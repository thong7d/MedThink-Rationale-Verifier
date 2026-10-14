import torch
import json
import copy
import re
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
from torch import nn
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class JointEncoder(T5Stack):
    def __init__(self, encoder_config, embed_tokens, patch_size=(100, 256)):
        super().__init__(encoder_config)
        self.set_input_embeddings(embed_tokens)
        self.model_dim = encoder_config.d_model

        self.mha_layer = nn.MultiheadAttention(
            embed_dim=self.model_dim, num_heads=encoder_config.num_heads,
            dropout=encoder_config.dropout_rate, batch_first=True
        )
        self.image_dense = nn.Linear(patch_size[1], self.model_dim)
        self.gate_dense = nn.Linear(2 * self.model_dim, self.model_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, image_ids=None, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict_actual = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict_actual,
        )
        hidden_states = outputs.last_hidden_state

        if image_ids is not None:
            if image_ids.dim() > 3:
                image_ids = image_ids.view(image_ids.size(0), -1, image_ids.size(-1))
            
            if image_ids.size(0) != hidden_states.size(0):
                beam_size = hidden_states.size(0) // image_ids.size(0)
                if beam_size > 0:
                    image_ids = image_ids.repeat_interleave(beam_size, dim=0)

            image_embedding = self.image_dense(image_ids)
            image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)
            merge = torch.cat([hidden_states, image_att], dim=-1)
            gate = self.sigmoid(self.gate_dense(merge))
            hidden_states = (1 - gate) * hidden_states + gate * image_att

        if not return_dict_actual:
            return tuple(v for v in [hidden_states, outputs.past_key_values, outputs.hidden_states, outputs.attentions] if v is not None)
            
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, patch_size=(100, 256)):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = JointEncoder(encoder_config, self.shared, patch_size)

    def forward(self, input_ids=None, image_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
                image_ids=image_ids, head_mask=head_mask, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=True,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

import torch
tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-base")
model = T5ForMultimodalGeneration.from_pretrained("allenai/unifiedqa-t5-base", attn_implementation="sdpa").to("cuda" if torch.cuda.is_available() else "cpu")
model.train()

inp = torch.randint(0, 1000, (4, 512)).to(model.device)
att = torch.ones(4, 512).to(model.device)
imgs = torch.randn(4, 100, 256).to(model.device)
lbls = torch.randint(0, 1000, (4, 512)).to(model.device)

print(f"inp: {inp.shape}, att: {att.shape}, imgs: {imgs.shape}, lbls: {lbls.shape}")

out = model(input_ids=inp, attention_mask=att, image_ids=imgs, labels=lbls)
print("Forward success!")
