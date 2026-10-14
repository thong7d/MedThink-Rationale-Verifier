"""
model.py — T5ForMultimodalGeneration with JointEncoder (Gated Fusion).
Directly from MedThink paper (Gai et al., NAACL 2025).
This is a clean copy of the original model.py with minor fixes for compatibility.
"""

from transformers import T5Config, T5ForConditionalGeneration
try:
    from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG, T5Block, T5LayerNorm
except ImportError:
    from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm
    __HEAD_MASK_WARNING_MSG = "head_mask warning"
import copy
from transformers.modeling_outputs import (
    ModelOutput, BaseModelOutput, BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
)
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import os
import warnings
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
try:
    from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
except ImportError:
    assert_device_map, get_device_map = None, None
from torch.utils.checkpoint import checkpoint


class JointEncoder(T5Stack):
    """
    Modified T5 encoder with cross-attention + gated fusion for image features.
    
    Architecture:
        Text → T5 Encoder → hidden_states
        Image → Linear projection → image_embedding
        Cross-attention(hidden_states, image_embedding) → image_att
        Gated fusion: gate = sigmoid(W[hidden; image_att])
        output = (1-gate)*hidden + gate*image_att
    """
    def __init__(self, config, embed_tokens=None, patch_size=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.patch_num, self.patch_dim = patch_size
        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_size, kdim=config.hidden_size,
            vdim=config.hidden_size, num_heads=1, batch_first=True
        )
        self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0))
             for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.post_init()
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask: {attention_mask.shape}")
        if dtype is None:
            dtype = getattr(self, 'dtype', torch.float32)
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for encoder_attention_mask: {encoder_attention_mask.shape}")
        dtype = getattr(self, 'dtype', torch.float32)
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min
        return encoder_extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            dtype = getattr(self, 'dtype', torch.float32)
            head_mask = head_mask.to(dtype=dtype)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            image_ids=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))
                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # ---- Visual Grounding via Gated Fusion ----
        if image_ids.dim() > 3:
            image_ids = image_ids.view(image_ids.size(0), -1, image_ids.size(-1))

        image_embedding = self.image_dense(image_ids)

        if image_embedding.dim() > 3:
            image_embedding = image_embedding.view(image_embedding.size(0), -1, image_embedding.size(-1))

        # Cross-attention: text queries attend to image keys/values
        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)

        # Gated fusion
        merge = torch.cat([hidden_states, image_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        hidden_states = (1 - gate) * hidden_states + gate * image_att

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states, present_key_value_states,
                    all_hidden_states, all_attentions, all_cross_attentions,
                ] if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    """
    T5 model extended with multimodal (vision) input via DETR features.
    
    The encoder is replaced with JointEncoder which fuses text and image
    features using cross-attention + gated fusion mechanism.
    
    Args:
        config: T5Config
        patch_size: tuple (num_patches, patch_dim) — typically (100, 256) for DETR
    """
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = JointEncoder(encoder_config, self.shared, patch_size)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)
        self.decoder.embed_tokens = self.shared

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            image_ids=None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                image_ids=image_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # Fix attention mask shape mismatch (encoder hidden may be longer than input mask)
        if attention_mask is not None and hidden_states is not None:
            if attention_mask.shape[1] != hidden_states.shape[1]:
                diff = hidden_states.shape[1] - attention_mask.shape[1]
                if diff > 0:
                    extra_mask = torch.ones(
                        (attention_mask.shape[0], diff),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, extra_mask], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, decoder_input_ids, past=None, attention_mask=None,
            use_cache=None, encoder_outputs=None, **kwargs
    ):
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # Fix attention mask vs encoder output shape mismatch during generation
        if encoder_outputs is not None and attention_mask is not None:
            enc_last_hs = getattr(
                encoder_outputs, 'last_hidden_state',
                encoder_outputs[0] if isinstance(encoder_outputs, (list, tuple)) else None
            )
            if enc_last_hs is not None and attention_mask.shape[1] != enc_last_hs.shape[1]:
                diff = enc_last_hs.shape[1] - attention_mask.shape[1]
                if diff > 0:
                    extra_mask = torch.ones(
                        (attention_mask.shape[0], diff),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, extra_mask], dim=1)

        output = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if "image_ids" in kwargs:
            output["image_ids"] = kwargs["image_ids"]

        return output
