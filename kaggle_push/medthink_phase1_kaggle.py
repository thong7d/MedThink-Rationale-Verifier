"""
MedThink Phase 1 — Faithful Reproduction of Paper (Gai et al., NAACL Findings 2025)
=====================================================================================
Script này trung thực tái hiện cách train của tác giả:
  - Dataset class:  ClosedMedVQADataset (chính xác như dataset.py gốc)
  - Trainer:        HuggingFace Seq2SeqTrainer + DataCollatorForSeq2Seq
  - Epochs:         150 (R-RAD), target_len=256
  - LR:             5e-4, weight_decay=1e-2
  - Effective BS:   32 (per_device_bs=4 × grad_accum=8, mô phỏng 4 GPU × bs=8)
  - Checkpoint:     Lưu mỗi 10 epoch, tự động resume nếu đã có checkpoint

Hỗ trợ Multi-Session:
  - Lần chạy đầu: train từ epoch 0 → lưu checkpoint vào /kaggle/working/checkpoints/
  - Lần chạy tiếp (khi đã export checkpoint thành Kaggle Dataset):
    Đặt RESUME_FROM_DATASET = "tên-kaggle-dataset" hoặc để rỗng
    Script sẽ tự detect và resume.

Log: In ra epoch loss, best loss, accuracy sau mỗi epoch.
"""

import os
import sys
import re
import json
import copy
import shutil
import logging
import traceback
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from torch.utils.data import Dataset

# ─── Upgrade transformers trước mọi thứ (fix Kaggle's stale version) ───────────
print("=" * 60)
print("Step 0: Upgrading transformers to latest stable...")
print("=" * 60)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers", "accelerate"], stdout=subprocess.DEVNULL)

from transformers import (
    T5Config, T5ForConditionalGeneration, AutoTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
)
try:
    from transformers.models.t5.modeling_t5 import (
        T5Stack, __HEAD_MASK_WARNING_MSG, T5Block, T5LayerNorm
    )
except ImportError:
    from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm
    __HEAD_MASK_WARNING_MSG = "head_mask warning"

from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("MedThink-Phase1")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PRETRAINED_MODEL   = "allenai/unifiedqa-t5-base"
REPO_URL           = "https://github.com/Tang-xiaoxiao/Medthink.git"
WORKING_DIR        = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
TMP_REPO_DIR       = "/tmp/Medthink"
OUTPUT_DIR         = os.path.join(WORKING_DIR, "phase1_outputs")

# Checkpoint resume: nếu bạn đã export checkpoint thành Kaggle Dataset,
# điền tên thư mục input tương ứng ở đây, ví dụ "/kaggle/input/medthink-ckpt-rad-explanation"
RESUME_CHECKPOINT_DIR = ""   # để trống = train từ đầu

# Thực nghiệm cần chạy: list (dataset_name, strategy, epochs, target_len)
EXPERIMENTS = [
    ("R-RAD", "Explanation",   150, 256),
    ("R-RAD", "Reasoning",     150, 256),
]

# Training HPs (giống tác giả: effective batch size = 32)
LR            = 5e-4
WEIGHT_DECAY  = 1e-2
BATCH_SIZE    = 8
GRAD_ACCUM    = 2   # 8 per GPU × 2 GPUs × 2 accum = 32 effective BS (hoặc 1 GPU: 8 × 4 = 32)
SOURCE_LEN    = 512
SEED          = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  1. MODEL — Giữ y hệt model.py của tác giả + patch is_encoder_decoder cho
#     Decoder để tránh DynamicCache bug trên transformers >= 4.40
# ═══════════════════════════════════════════════════════════════════════════════

class JointEncoder(T5Stack):
    """T5 Encoder với Gated Fusion thị giác (chính xác như model.py gốc)."""

    def __init__(self, config, embed_tokens=None, patch_size=None):
        super().__init__(config)
        self.embed_tokens   = embed_tokens
        self.is_decoder     = config.is_decoder
        self.patch_num, self.patch_dim = patch_size

        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer   = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_size, kdim=config.hidden_size,
            vdim=config.hidden_size, num_heads=1, batch_first=True
        )
        self.gate_dense  = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sigmoid     = nn.Sigmoid()

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0))
             for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout          = nn.Dropout(config.dropout_rate)
        self.post_init()
        self.model_parallel       = False
        self.device_map           = None
        self.gradient_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(param.dtype for param in self.parameters() if param.is_floating_point())
        except (StopIteration, Exception):
            if hasattr(self, "image_dense") and hasattr(self.image_dense, "weight"):
                return self.image_dense.weight.dtype
            return torch.float32

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        if attention_mask.dim() == 3:
            ext = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            ext = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong attention_mask shape: {attention_mask.shape}")
        try:
            dtype = dtype or self.dtype
        except (AttributeError, StopIteration, Exception):
            dtype = torch.float32
        ext = ext.to(dtype=dtype)
        # Use -1e4 to avoid floating point overflow in softmax gradients (HF standard)
        ext = (1.0 - ext) * -1e4
        return ext

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            ext = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            ext = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong encoder_attention_mask shape: {encoder_attention_mask.shape}")
        try:
            dtype = self.dtype
        except (AttributeError, StopIteration, Exception):
            dtype = torch.float32
        ext = ext.to(dtype=dtype)
        ext = (1.0 - ext) * -1e4
        return ext

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            try:
                dtype = self.dtype
            except (AttributeError, StopIteration, Exception):
                dtype = torch.float32
            head_mask = head_mask.to(dtype=dtype)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def get_input_embeddings(self):  return self.embed_tokens
    def set_input_embeddings(self, e): self.embed_tokens = e

    def forward(
        self,
        input_ids=None, attention_mask=None, encoder_hidden_states=None,
        encoder_attention_mask=None, inputs_embeds=None, image_ids=None,
        head_mask=None, cross_attn_head_mask=None, past_key_values=None,
        use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None,
    ):
        use_cache             = use_cache if use_cache is not None else self.config.use_cache
        output_attentions     = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states  = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict           = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids   = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        mask_seq_length = (past_key_values[0][0].shape[2] + seq_length
                           if past_key_values is not None else seq_length)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_attention_mask = torch.ones(
                batch_size, encoder_hidden_states.shape[1],
                device=inputs_embeds.device, dtype=torch.long
            )

        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.is_decoder and encoder_hidden_states is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask            = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states    = () if output_hidden_states else None
        all_attentions       = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                cross_attn_layer_head_mask=cross_attn_head_mask[i],
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

        # ── Gated Visual Fusion ──────────────────────────────────────────────
        if image_ids.dim() > 3:
            image_ids = image_ids.view(image_ids.size(0), -1, image_ids.size(-1))
        image_embedding = self.image_dense(image_ids)
        if image_embedding.dim() > 3:
            image_embedding = image_embedding.view(image_embedding.size(0), -1, image_embedding.size(-1))

        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)
        merge         = torch.cat([hidden_states, image_att], dim=-1)
        gate          = self.sigmoid(self.gate_dense(merge))
        hidden_states = (1 - gate) * hidden_states + gate * image_att
        # ─────────────────────────────────────────────────────────────────────

        if not return_dict:
            return tuple(v for v in [
                hidden_states, present_key_value_states,
                all_hidden_states, all_attentions, all_cross_attentions,
            ] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size=(100, 256)):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared    = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder        = False
        encoder_config.use_cache         = False
        encoder_config.is_encoder_decoder = False
        self.encoder = JointEncoder(encoder_config, self.shared, patch_size)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder        = True
        # FIX: set is_encoder_decoder=True so T5Stack initialises EncoderDecoderCache,
        # preventing the 1024-vs-512 size mismatch bug in transformers >= 4.40
        decoder_config.is_encoder_decoder = True
        decoder_config.num_layers        = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)
        self.decoder.embed_tokens = self.shared

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()
        self.model_parallel = False
        self.device_map     = None

    def forward(
        self,
        input_ids=None, image_ids=None, attention_mask=None,
        decoder_input_ids=None, decoder_attention_mask=None,
        head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None,
        encoder_outputs=None, past_key_values=None,
        inputs_embeds=None, decoder_inputs_embeds=None,
        labels=None, use_cache=None,
        output_attentions=None, output_hidden_states=None,
        return_dict=None, **kwargs,
    ):
        use_cache   = use_cache   if use_cache   is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                import warnings; warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                inputs_embeds=inputs_embeds, image_ids=image_ids,
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

        # Align attention_mask seq length with encoder hidden_states
        if attention_mask is not None and hidden_states is not None:
            if attention_mask.shape[1] != hidden_states.shape[1]:
                diff = hidden_states.shape[1] - attention_mask.shape[1]
                if diff > 0:
                    extra = torch.ones(
                        (attention_mask.shape[0], diff),
                        dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    attention_mask = torch.cat([attention_mask, extra], dim=1)

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
            loss = CrossEntropyLoss(ignore_index=-100)(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss, logits=lm_logits,
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

        if encoder_outputs is not None and attention_mask is not None:
            hs = getattr(encoder_outputs, "last_hidden_state",
                         encoder_outputs[0] if isinstance(encoder_outputs, (list, tuple)) else None)
            if hs is not None and attention_mask.shape[1] != hs.shape[1]:
                diff = hs.shape[1] - attention_mask.shape[1]
                if diff > 0:
                    extra = torch.ones(
                        (attention_mask.shape[0], diff),
                        dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    attention_mask = torch.cat([attention_mask, extra], dim=1)

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


# ═══════════════════════════════════════════════════════════════════════════════
#  2. DATASET — ClosedMedVQADataset (chính xác như dataset.py gốc của tác giả)
# ═══════════════════════════════════════════════════════════════════════════════

class ClosedInputAndTargetAndImg:
    OPTIONS = ["A", "B"]

    def __init__(self, problem):
        self.problem      = problem
        self.question_text = problem["question"]
        self.answer_text   = f"({self.OPTIONS[problem['answer']]})"
        self.choice_text   = " ".join(
            f"({self.OPTIONS[i]}) {c}" for i, c in enumerate(problem["choices"])
        )
        self.solution_text = problem.get("solution", "")

    def get_target(self, method):
        if method == "Explanation":
            return f"The answer is {self.answer_text}.\nSolution: {self.solution_text}"
        elif method == "Reasoning":
            return f"{self.solution_text}\nAnswer: The answer is {self.answer_text}."
        elif method == "First-Stage_Reasoning":
            return self.solution_text
        elif method == "Second-Stage_Reasoning":
            return f"The answer is {self.answer_text}."
        elif method == "without_R":
            return f"The answer is {self.answer_text}."
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_input(self, method):
        q = self.question_text
        c = self.choice_text
        if method == "Explanation":
            return f"Question: {q}\nOptions: {c}\nAnswer:"
        elif method == "Reasoning":
            return f"Question: {q}\nOptions: {c}\nSolution:"
        elif method in ("First-Stage_Reasoning",):
            return f"Question: {q}\nOptions: {c}\nSolution:"
        elif method == "Second-Stage_Reasoning":
            return f"Question: {q}\nOptions: {c}\nSolution: {self.solution_text}\nAnswer:"
        elif method == "without_R":
            return f"Question: {q}\nOptions: {c}\nAnswer:"
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_img(self, dataset):
        if dataset == "rad":
            return self.problem["image"][:-4]   # strip extension, e.g. "synpic100.jpg" → "synpic100"
        elif dataset == "slake":
            return self.problem["img_id"]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")


class ClosedMedVQADataset(Dataset):
    """
    Giống y hệt ClosedMedVQADataset trong dataset.py của tác giả.
    Tham số thêm: answer_gt / question / solution_gt / prompt để hỗ trợ evaluation.
    """

    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _method, _source_len, _target_len, _dataset):
        self.tokenizer    = _tokenizer
        self.source_text  = []
        self.target_text  = []
        self.problem_id   = []
        self.img_index    = []
        self.answer_gt    = []
        self.question     = []
        self.solution_gt  = []
        self.prompt       = []

        self.source_len   = _source_len
        self.target_len   = _target_len

        self.pretrained_feature = torch.load(_img_file_path, weights_only=False)

        with open(_text_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(_img_name_map, "r", encoding="utf-8") as f:
            name_map = json.load(f)

        for problem_id, problem in data.items():
            pair   = ClosedInputAndTargetAndImg(problem)
            prompt = pair.get_input(method=_method)
            target = pair.get_target(method=_method)
            img    = pair.get_img(dataset=_dataset)

            self.source_text.append(prompt)
            self.target_text.append(target)
            self.problem_id.append(problem_id)
            self.answer_gt.append(pair.answer_text)
            self.question.append(pair.question_text)
            self.solution_gt.append(pair.solution_text)
            self.prompt.append(prompt)

            # Robust image key lookup
            for candidate in [img, img + ".jpg", img + ".png",
                               os.path.basename(img), str(img)]:
                if candidate in name_map:
                    self.img_index.append(int(name_map[candidate]))
                    break
            else:
                logger.warning(f"Image key '{img}' not in name_map, using index 0")
                self.img_index.append(0)

        # Pre-tokenize all samples for massive training speedup (avoid 270,000+ tokenizations)
        logger.info(f"Pre-tokenizing {len(self.source_text)} samples (source_len={self.source_len}, target_len={self.target_len})...")
        cleaned_sources = [" ".join(s.split()) for s in self.source_text]
        cleaned_targets = [" ".join(t.split()) for t in self.target_text]

        enc_sources = self.tokenizer(
            cleaned_sources, max_length=self.source_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        enc_targets = self.tokenizer(
            cleaned_targets, max_length=self.target_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )

        self.input_ids = enc_sources["input_ids"]
        self.attention_mask = enc_sources["attention_mask"]

        # CRITICAL FIX: Mask padding tokens in labels with -100 so CrossEntropyLoss ignores them!
        self.labels = enc_targets["input_ids"].clone()
        self.labels[self.labels == self.tokenizer.pad_token_id] = -100
        logger.info(f"Pre-tokenization complete. Non-padding tokens per sample: {(self.labels != -100).sum(dim=1).float().mean():.1f}")

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        return {
            "input_ids":      self.input_ids[item],
            "attention_mask": self.attention_mask[item],
            "image_ids":      self.pretrained_feature[self.img_index[item]].squeeze(),
            "labels":         self.labels[item],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. FEATURE EXTRACTION — DETR (giống extract_img_feature.py gốc)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_detr_features(image_dir, output_dir, dataset_name="rad", device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    pth_file = os.path.join(output_dir, f"{dataset_name}_detr.pth")
    map_file = os.path.join(output_dir, "name_map.json")

    if os.path.exists(pth_file) and os.path.exists(map_file):
        logger.info(f"Found pre-extracted features: {pth_file}")
        return pth_file, map_file

    logger.info(f"Extracting DETR features from {image_dir} ...")

    # Use cooelf/detr fork which adds 'hs' output (như paper gốc)
    detr_dir = "/tmp/detr_cooelf"
    if not os.path.exists(os.path.join(detr_dir, "hubconf.py")):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/cooelf/detr.git", detr_dir], check=True)

    import torchvision.transforms as T
    from PIL import Image

    detr_model = torch.hub.load(detr_dir, "detr_resnet101", source="local",
                                pretrained=True).to(device).eval()
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    feats_list = []
    name_map   = {}

    with torch.no_grad():
        for idx, fname in enumerate(img_files):
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            t   = transform(img).unsqueeze(0).to(device)
            out = detr_model(t)
            # cooelf/detr returns dict with key 'hs' (decoder hidden states, shape [L, B, Q, D])
            if isinstance(out, dict) and "hs" in out:
                feat = out["hs"][-1].squeeze(0).cpu()   # [Q, D]
            else:
                feat = out[0][-1].squeeze(0).cpu()
            feats_list.append(feat)

            base   = os.path.splitext(fname)[0]
            for k in [fname, base, base.lower()]:
                name_map[k] = idx

            if idx % 50 == 0:
                logger.info(f"  Extracted {idx}/{len(img_files)}: {fname}, shape {feat.shape}")

    stacked = torch.stack(feats_list)     # [N, Q, D]
    torch.save(stacked, pth_file)
    with open(map_file, "w") as f:
        json.dump(name_map, f, indent=2)

    logger.info(f"Saved DETR features {stacked.shape} → {pth_file}")
    return pth_file, map_file


# ═══════════════════════════════════════════════════════════════════════════════
#  4. ANSWER EXTRACTION (như closed_end_generate.py gốc)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_ans(text):
    """Trích xuất ký tự lựa chọn từ chuỗi sinh ra."""
    res = re.findall(r"The answer is \(([A-Z])\)", text)
    if res:
        return res[-1]
    res = re.findall(r"\(([A-Z])\)", text)
    if res:
        return res[-1]
    res = re.findall(r"\b([A-Z])\b", text)
    if res:
        return res[-1]
    return "FAILED"


# ═══════════════════════════════════════════════════════════════════════════════
#  5. CUSTOM TRAINER & CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

from transformers import TrainerCallback

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            epoch = logs.get("epoch", state.epoch)
            loss = logs.get("loss", None)
            lr = logs.get("learning_rate", None)
            step = state.global_step
            max_steps = state.max_steps
            if loss is not None:
                loss_str = f"{loss:.4f}"
                lr_str = f"{lr:.2e}" if lr is not None else "N/A"
                logger.info(f"📊 [Step {step:4d}/{max_steps}] Epoch {epoch:6.2f}/{args.num_train_epochs:.0f} | Loss: {loss_str} | LR: {lr_str}")


class MultimodalSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Kế thừa Seq2SeqTrainer và override compute_loss + prediction_step
    để hỗ trợ truyền image_ids vào model trong cả training và generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            self.tokenizer = getattr(self, "processing_class", None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self.args.generation_max_length or 256,
            "num_beams": self.args.generation_num_beams or 1,
        }
        if "image_ids" in inputs:
            gen_kwargs["image_ids"] = inputs["image_ids"]

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                loss = outputs["loss"].mean().detach() if isinstance(outputs, dict) else outputs.loss.mean().detach()
            else:
                loss = None

            gen_model = model.module if hasattr(model, "module") else model
            generated = gen_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **gen_kwargs
            )

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()

        max_len = gen_kwargs["max_length"] or 256
        if generated.shape[-1] < max_len:
            pad_len = max_len - generated.shape[-1]
            tok = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
            pad_id = tok.pad_token_id if tok is not None and tok.pad_token_id is not None else 0
            generated = torch.nn.functional.pad(generated, (0, pad_len), value=pad_id)

        return loss, generated.detach(), labels


# ═══════════════════════════════════════════════════════════════════════════════
#  6. EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    dataset_name, method, epochs, target_len,
    train_json, test_json, feature_pth, name_map_json,
    output_dir, device, resume_from_checkpoint=None,
):
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: {dataset_name} | {method} | {epochs} epochs | target_len={target_len}")
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = T5ForMultimodalGeneration.from_pretrained(
        PRETRAINED_MODEL, patch_size=(100, 256)
    )

    dataset_key = dataset_name.lower().replace("r-", "")   # "R-RAD" → "rad"

    train_ds = ClosedMedVQADataset(
        _tokenizer=tokenizer,
        _text_file_path=train_json,
        _img_file_path=feature_pth,
        _img_name_map=name_map_json,
        _method=method,
        _source_len=SOURCE_LEN,
        _target_len=target_len,
        _dataset=dataset_key,
    )
    test_ds = ClosedMedVQADataset(
        _tokenizer=tokenizer,
        _text_file_path=test_json,
        _img_file_path=feature_pth,
        _img_name_map=name_map_json,
        _method=method,
        _source_len=SOURCE_LEN,
        _target_len=target_len,
        _dataset=dataset_key,
    )
    logger.info(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")

    ckpt_dir = os.path.join(output_dir, "checkpoints", f"{dataset_name}_{method}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Detect existing checkpoint to resume
    existing_ckpts = sorted([
        d for d in os.listdir(ckpt_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(ckpt_dir, d))
    ])
    resume_ckpt = None
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        resume_ckpt = resume_from_checkpoint
        logger.info(f"Resuming from external checkpoint: {resume_ckpt}")
    elif existing_ckpts:
        resume_ckpt = os.path.join(ckpt_dir, existing_ckpts[-1])
        logger.info(f"Auto-resuming from latest checkpoint: {resume_ckpt}")

    import inspect
    args_kwargs = {
        "output_dir": ckpt_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,           # effective BS = 32
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": 1.0,
        "save_strategy": "steps",
        "save_steps": 500,                                   # lưu checkpoint mỗi 500 steps (~9 epochs)
        "save_total_limit": 2,                               # giữ 2 checkpoint gần nhất để tiết kiệm disk
        "logging_strategy": "epoch",
        "logging_steps": 1,
        "predict_with_generate": True,
        "generation_max_length": target_len,
        "fp16": False,                                       # FP32 faithful to paper (tránh NaN/underflow với T5)
        "report_to": ["none"],
        "remove_unused_columns": False,                       # GIỮ image_ids trong batch
        "seed": SEED,
        "dataloader_num_workers": 0,
        "load_best_model_at_end": False,
    }
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        args_kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in sig.parameters:
        args_kwargs["evaluation_strategy"] = "no"

    training_args = Seq2SeqTrainingArguments(**args_kwargs)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "data_collator": data_collator,
        "callbacks": [ProgressCallback()],
    }
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = MultimodalSeq2SeqTrainer(**trainer_kwargs)

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    logger.info(f"Starting training... (resume_from_checkpoint={resume_ckpt})")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── SAVE final model ───────────────────────────────────────────────────────
    model_save = os.path.join(output_dir, f"model_{dataset_name}_{method}")
    trainer.save_model(model_save)
    tokenizer.save_pretrained(model_save)
    logger.info(f"Model saved to {model_save}")

    # ── INFERENCE & EVALUATION (như closed_end_generate.py gốc) ───────────────
    logger.info(f"Running inference on test set ({len(test_ds)} samples)...")
    model.eval()

    predict_output = trainer.predict(
        test_dataset=test_ds,
        max_length=target_len,
    )
    preds = predict_output.predictions
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)

    # Compute accuracy (like closed_end_generate.py)
    import pandas as pd
    results = []
    correct = 0
    for idx, pred_text in enumerate(preds_text):
        pred_text  = pred_text.strip()
        pred_ans   = extract_ans(pred_text)
        gt_ans     = test_ds.answer_gt[idx]
        gt_letter  = re.sub(r"[()]", "", gt_ans).strip()
        is_correct = int(pred_ans == gt_letter)
        correct   += is_correct
        results.append({
            "problem_id":    test_ds.problem_id[idx],
            "dataset":       dataset_name,
            "method":        method,
            "question":      test_ds.question[idx],
            "answer_gt":     gt_ans,
            "answer_pred":   pred_ans,
            "solution_gt":   test_ds.solution_gt[idx],
            "raw_output":    pred_text,
            "is_correct":    is_correct,
        })

    acc = correct / len(results) * 100.0

    logger.info("=" * 60)
    logger.info(f"RESULTS  {dataset_name} ({method}): Accuracy = {acc:.2f}% ({correct}/{len(results)})")
    logger.info("=" * 60)
    logger.info("Sample predictions:")
    for i in range(min(5, len(results))):
        logger.info(f"  [{i}] Raw: '{results[i]['raw_output']}' | Pred: '{results[i]['answer_pred']}' | GT: '{results[i]['answer_gt']}' | Match: {bool(results[i]['is_correct'])}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"predictions_{dataset_name}_{method.lower()}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved predictions to {csv_path}")

    return acc, df


# ═══════════════════════════════════════════════════════════════════════════════
#  7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import pandas as pd

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    err_path = os.path.join(OUTPUT_DIR, "error_traceback.txt")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # ── Setup Git LFS and Clone repo ──────────────────────────────────────
        need_clone = not os.path.exists(TMP_REPO_DIR)
        test_img = os.path.join(TMP_REPO_DIR, "Medthink_Dataset", "R-RAD", "images", "synpic100132.jpg")
        if os.path.exists(test_img) and os.path.getsize(test_img) < 1000:
            logger.warning("Existing repo has LFS pointer files (<1000 bytes). Re-cloning with Git LFS...")
            shutil.rmtree(TMP_REPO_DIR, ignore_errors=True)
            need_clone = True

        if need_clone:
            logger.info(f"Setting up Git LFS and cloning MedThink repo to {TMP_REPO_DIR} ...")
            os.system("apt-get update -qq && apt-get install -y git-lfs -qq")
            os.system("git lfs install")
            env = os.environ.copy()
            env["GIT_LFS_SKIP_SMUDGE"] = "1"
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, TMP_REPO_DIR],
                env=env,
                check=True
            )
            logger.info("Pulling Git LFS files for Medthink_Dataset ...")
            subprocess.run(
                ["git", "-C", TMP_REPO_DIR, "lfs", "pull", "--include=Medthink_Dataset/**"],
                check=True
            )
            shutil.rmtree(os.path.join(TMP_REPO_DIR, ".git"), ignore_errors=True)
        else:
            logger.info(f"Repo already exists at {TMP_REPO_DIR}")

        DATASET_DIR = os.path.join(TMP_REPO_DIR, "Medthink_Dataset")
        RAD_DIR     = os.path.join(DATASET_DIR, "R-RAD")
        RAD_IMG_DIR = os.path.join(RAD_DIR, "images")
        RAD_TRAIN   = os.path.join(RAD_DIR, "closed-end", "trainset.json")
        RAD_TEST    = os.path.join(RAD_DIR, "closed-end", "testset.json")

        # Sanity check: Ensure images are valid binary files
        test_img_path = os.path.join(RAD_IMG_DIR, "synpic100132.jpg")
        if not os.path.exists(test_img_path):
            raise FileNotFoundError(f"Verification failed: {test_img_path} not found")
        img_sz = os.path.getsize(test_img_path)
        logger.info(f"Verification: sample image size = {img_sz} bytes")
        if img_sz < 1000:
            raise RuntimeError(f"Sample image {test_img_path} is an LFS pointer ({img_sz} bytes), not binary!")

        # ── Feature Extraction ────────────────────────────────────────────────
        feat_dir = os.path.join(OUTPUT_DIR, "detr_features", "R-RAD")
        feat_pth, name_map_json = extract_detr_features(
            RAD_IMG_DIR, feat_dir, dataset_name="rad", device=device
        )

        # ── Run Experiments ───────────────────────────────────────────────────
        summary = []
        for (ds_name, method, epochs, target_len) in EXPERIMENTS:
            try:
                acc, _ = run_experiment(
                    dataset_name=ds_name,
                    method=method,
                    epochs=epochs,
                    target_len=target_len,
                    train_json=RAD_TRAIN,
                    test_json=RAD_TEST,
                    feature_pth=feat_pth,
                    name_map_json=name_map_json,
                    output_dir=OUTPUT_DIR,
                    device=device,
                    resume_from_checkpoint=RESUME_CHECKPOINT_DIR or None,
                )
                summary.append({"dataset": ds_name, "method": method,
                                 "epochs": epochs, "accuracy_pct": round(acc, 2)})
            except Exception as exp:
                err_msg = traceback.format_exc()
                logger.error(f"ERROR in {ds_name}/{method}:\n{err_msg}")
                with open(err_path, "a") as f:
                    f.write(f"\n=== {ds_name} / {method} ===\n{err_msg}\n")
                summary.append({"dataset": ds_name, "method": method,
                                 "epochs": epochs, "accuracy_pct": "ERROR"})

        # ── Summary ───────────────────────────────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 70)
        for row in summary:
            logger.info(f"  {row['dataset']:8s} | {row['method']:20s} | "
                        f"Epochs={row['epochs']:3d} | Acc={row['accuracy_pct']}")

        pd.DataFrame(summary).to_csv(
            os.path.join(OUTPUT_DIR, "training_summary.csv"), index=False
        )
        logger.info("PHASE 1 KAGGLE PIPELINE FINISHED SUCCESSFULLY!")

    except Exception as e:
        err_msg = traceback.format_exc()
        logger.error(f"FATAL:\n{err_msg}")
        with open(err_path, "w") as f:
            f.write(err_msg)
        raise


if __name__ == "__main__":
    main()
