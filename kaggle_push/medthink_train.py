# ============================================================
# Cell 6: Self-Healed Stage 1 Training (First-Stage_Reasoning, LR=5e-4, 25 Epochs)
# ============================================================
import os, sys, json, importlib, copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torchvision.transforms as T
from PIL import Image
from tqdm.notebook import tqdm
import transformers
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from torch.utils.data import Dataset

PROJECT_ROOT = '/kaggle/working/MedThink_Verifier'
OUTPUT_DIR   = f'{PROJECT_ROOT}/phase1_predictions'
CKPT_DIR     = f'{PROJECT_ROOT}/checkpoints'
REPO_DIR     = f'{PROJECT_ROOT}/Medthink_repo'
CODE_DIR     = f'{REPO_DIR}/Medthink_Code'
DATA_DIR     = '/kaggle/input/medthink-dataset/Medthink-Dataset'

for d in [OUTPUT_DIR, CKPT_DIR, CODE_DIR, f'{PROJECT_ROOT}/features/R-RAD', f'{PROJECT_ROOT}/features/R-SLAKE']:
    os.makedirs(d, exist_ok=True)

detr_pth = f'{PROJECT_ROOT}/features/R-RAD/detr.pth'
name_map = f'{PROJECT_ROOT}/features/R-RAD/name_map.json'

# Auto extract features if missing
if not os.path.exists(detr_pth) or os.path.getsize(detr_pth) < 1000:
    print("🔄 Visual feature file missing. Extracting DETR features for R-RAD...")
    rad_img_dir = f'{DATA_DIR}/R-RAD/images'
    if not os.path.exists(rad_img_dir) or len(os.listdir(rad_img_dir)) == 0:
        os.system('apt-get install -y git-lfs -q')
        os.system('git lfs install')
        tmp_clone = '/content/medthink_tmp'
        os.system(f'rm -rf {tmp_clone}')
        os.system(f'GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Tang-xiaoxiao/Medthink.git {tmp_clone}')
        os.system(f'cd {tmp_clone} && git lfs pull --include="Medthink_Dataset/**"')
        os.makedirs(REPO_DIR, exist_ok=True)
        os.system(f'cp -r {tmp_clone}/Medthink_Dataset {REPO_DIR}/')
        os.system(f'rm -rf {tmp_clone}')

    detr_dir = f'{PROJECT_ROOT}/detr_cooelf'
    if not os.path.exists(f'{detr_dir}/hubconf.py'):
        os.system(f'git clone --depth 1 https://github.com/cooelf/detr.git {detr_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detr_model = torch.hub.load(detr_dir, 'detr_resnet101', source='local', pretrained=True).to(device).eval()
    transform = T.Compose([T.Resize((800, 800)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img_files = [f for f in os.listdir(rad_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    feats_list, nmap = [], {}
    with torch.no_grad():
        for idx, fname in enumerate(tqdm(sorted(img_files), desc="Extracting RAD DETR")):
            img = Image.open(os.path.join(rad_img_dir, fname)).convert('RGB')
            t_img = transform(img).unsqueeze(0).to(device)
            out = detr_model(t_img)
            last_hs = out['hs'][-1].squeeze(0) if isinstance(out, dict) and 'hs' in out else out[0][-1].squeeze(0)
            feats_list.append(last_hs.cpu())
            bname = os.path.basename(fname)
            no_ext = os.path.splitext(bname)[0]
            nmap[fname] = idx
            nmap[bname] = idx
            nmap[no_ext] = idx
            nmap[str(no_ext)] = idx

    stacked = torch.stack(feats_list)
    torch.save(stacked, detr_pth)
    with open(name_map, 'w') as f:
        json.dump(nmap, f, indent=2)
    print(f"✅ Extracted and saved RAD DETR features {stacked.shape} to {detr_pth}")

# 1. Dataset class definition
class OpenMedVQADataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _method, _dataset, _source_len, _target_len):
        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        self.source_len = _source_len
        self.target_len = _target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)

        for problem in data:
            pair = OpenInputAndTargetAndImg(data[problem])
            prompt = pair.get_input(method=_method)
            target = pair.get_target(method=_method)
            img = pair.get_img(_dataset=_dataset)
            self.source_text.append(prompt)
            self.target_text.append(target)
            img_key = str(img)
            if img_key not in name_map:
                if f"{img_key}.jpg" in name_map:
                    img_key = f"{img_key}.jpg"
                elif f"{img_key}.png" in name_map:
                    img_key = f"{img_key}.png"
                elif os.path.splitext(img_key)[0] in name_map:
                    img_key = os.path.splitext(img_key)[0]
                else:
                    img_key = list(name_map.keys())[0]
            self.img_index.append(int(name_map[img_key]))
            self.problem_id.append(problem)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])
        img_index = self.img_index[item]

        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer(
            source_text,
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer(
            target_text,
            max_length=self.target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        image_ids = self.pretrained_feature[img_index].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids
        }

class OpenInputAndTargetAndImg:
    def __init__(self, problem):
        self.problem = problem
        self.question_text = self.get_question_text()
        self.answer_text = self.get_answer()
        self.solution_text = self.get_solution_text()

    def get_question_text(self):
        return self.problem['question']

    def get_answer(self):
        return self.problem['choices'][0]

    def get_solution_text(self):
        return self.problem.get('solution', '')

    def get_target(self, method):
        if method == "Explanation":
            return f"The answer is {self.answer_text}.\nSolution: {self.problem.get('solution', '')}"
        elif method == "Reasoning":
            return f"{self.problem.get('solution', '')}\nAnswer: The answer is {self.answer_text}."
        elif method == "First-Stage_Reasoning":
            return f"{self.problem.get('solution', '')}"
        elif method == "Second-Stage_Reasoning":
            return f"The answer is {self.answer_text}."
        elif method == "without_R":
            return f"The answer is {self.answer_text}."
        else:
            raise ValueError(f"Invalid method: {method}")

    def get_input(self, method):
        if method == "Explanation":
            return f"Question: {self.question_text}\nAnswer:"
        elif method == "Reasoning":
            return f"Question: {self.question_text}\nSolution:"
        elif method == "First-Stage_Reasoning":
            return f"Question: {self.question_text}\nSolution:"
        elif method == "Second-Stage_Reasoning":
            sol = self.solution_text if self.solution_text else ""
            return f"Question: {self.question_text}\nSolution: {sol}\nAnswer:"
        else:
            raise ValueError(f"Invalid method: {method}")

    def get_img(self, _dataset):
        if _dataset == "rad":
            return self.problem['image'][:-4]
        elif _dataset == "slake":
            return str(self.problem["img_id"])
        else:
            raise ValueError(f"Invalid dataset: {_dataset}")

# 2. JointEncoder & Model definition
class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None, patch_size=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.patch_num, self.patch_dim = patch_size
        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                     vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.post_init()

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

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, inputs_embeds=None, image_ids=None,
                head_mask=None, cross_attn_head_mask=None, past_key_values=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        hidden_states = self.dropout(inputs_embeds)

        position_bias = None
        for i, layer_module in enumerate(self.block):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                layer_head_mask=head_mask[i],
                use_cache=False,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if image_ids is not None:
            if image_ids.dim() > 3:
                image_ids = image_ids.view(image_ids.size(0), -1, image_ids.size(-1))
            image_embedding = self.image_dense(image_ids)
            if image_embedding.dim() > 3:
                image_embedding = image_embedding.view(image_embedding.size(0), -1, image_embedding.size(-1))
            image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)
            merge = torch.cat([hidden_states, image_att], dim=-1)
            gate = self.sigmoid(self.gate_dense(merge))
            hidden_states = (1 - gate) * hidden_states + gate * image_att

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)

class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    def __init__(self, config, patch_size=(100, 256)):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = JointEncoder(encoder_config, patch_size=patch_size)
        self.encoder.embed_tokens = self.shared
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)
        self.decoder.embed_tokens = self.shared
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, image_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, encoder_outputs=None, past_key_values=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_ids=image_ids,
            )

        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
        )

        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                     encoder_outputs=None, **kwargs):
        image_ids = kwargs.get("image_ids", None)
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "image_ids": image_ids,
        }

# 3. Execute Stage 1 Training
base_model_name = 'allenai/unifiedqa-t5-base'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

ds_name = 'rad'
method = 'First-Stage_Reasoning'
num_epochs = 25
learning_rate = 5e-4

print(f"============================================================")
print(f"🚀 Running Stage 1 Training ({method}) | Epochs: {num_epochs} | LR: {learning_rate}")
print(f"============================================================")

ckpt_out_dir = f'{CKPT_DIR}/{ds_name}_{method}'
os.makedirs(ckpt_out_dir, exist_ok=True)

train_json = f'{DATA_DIR}/R-RAD/open-end/trainset.json'

model_obj = T5ForMultimodalGeneration.from_pretrained(
    base_model_name,
    patch_size=(100, 256)
)

train_dataset = OpenMedVQADataset(
    _tokenizer=tokenizer,
    _text_file_path=train_json,
    _img_file_path=detr_pth,
    _img_name_map=name_map,
    _method=method,
    _dataset=ds_name,
    _source_len=512,
    _target_len=256
)

print(f"Train dataset size: {len(train_dataset)} samples")

training_args = Seq2SeqTrainingArguments(
    output_dir=ckpt_out_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=learning_rate,
    weight_decay=0.01,
    save_strategy='epoch',
    save_total_limit=1,
    eval_strategy='no',
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    report_to='none',
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    model=model_obj,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

print(f"🚀 Starting Stage 1 Training ({num_epochs} Epochs, LR={learning_rate})...")
train_result = trainer.train()
trainer.save_model(ckpt_out_dir)
print(f"✅ Stage 1 Training complete! Model saved to {ckpt_out_dir}")


# ============================================================
# Cell 8: Self-Healed Stage 2 Training (Second-Stage_Reasoning, LR=5e-5, 20 Epochs, target_len=32)
# ============================================================
import os, sys, json, importlib, copy, gc
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import transformers
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput

# Clean up memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

PROJECT_ROOT = '/kaggle/working/MedThink_Verifier'
OUTPUT_DIR   = f'{PROJECT_ROOT}/phase1_predictions'
CKPT_DIR     = f'{PROJECT_ROOT}/checkpoints'
REPO_DIR     = f'{PROJECT_ROOT}/Medthink_repo'
CODE_DIR     = f'{REPO_DIR}/Medthink_Code'
DATA_DIR     = '/kaggle/input/medthink-dataset/Medthink-Dataset'

aug_train_json = f'{OUTPUT_DIR}/augmented_trainset.json'
if not os.path.exists(aug_train_json):
    print("ℹ️ Augmented trainset not found, using base trainset.json for Stage 2...")
    aug_train_json = f'{DATA_DIR}/R-RAD/open-end/trainset.json'

detr_pth = f'{PROJECT_ROOT}/features/R-RAD/detr.pth'
name_map = f'{PROJECT_ROOT}/features/R-RAD/name_map.json'

class OpenMedVQADataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _method, _dataset, _source_len, _target_len):
        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        self.source_len = _source_len
        self.target_len = _target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)

        for problem in data:
            pair = OpenInputAndTargetAndImg(data[problem])
            prompt = pair.get_input(method=_method)
            target = pair.get_target(method=_method)
            img = pair.get_img(_dataset=_dataset)
            self.source_text.append(prompt)
            self.target_text.append(target)
            img_key = str(img)
            if img_key not in name_map:
                if f"{img_key}.jpg" in name_map:
                    img_key = f"{img_key}.jpg"
                elif f"{img_key}.png" in name_map:
                    img_key = f"{img_key}.png"
                elif os.path.splitext(img_key)[0] in name_map:
                    img_key = os.path.splitext(img_key)[0]
                else:
                    img_key = list(name_map.keys())[0]
            self.img_index.append(int(name_map[img_key]))
            self.problem_id.append(problem)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])
        img_index = self.img_index[item]

        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer(
            source_text,
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer(
            target_text,
            max_length=self.target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        image_ids = self.pretrained_feature[img_index].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids
        }

class OpenInputAndTargetAndImg:
    def __init__(self, problem):
        self.problem = problem
        self.question_text = self.get_question_text()
        self.answer_text = self.get_answer()
        self.solution_text = self.get_solution_text()

    def get_question_text(self):
        return self.problem['question']

    def get_answer(self):
        return self.problem['choices'][0]

    def get_solution_text(self):
        return self.problem.get('solution', '')

    def get_target(self, method):
        return f"The answer is {self.answer_text}."

    def get_input(self, method):
        sol = self.solution_text if self.solution_text else ""
        return f"Question: {self.question_text}\nSolution: {sol}\nAnswer:"

    def get_img(self, _dataset):
        if _dataset == "rad":
            return self.problem['image'][:-4]
        elif _dataset == "slake":
            return str(self.problem["img_id"])
        else:
            raise ValueError(f"Invalid dataset: {_dataset}")

class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None, patch_size=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.patch_num, self.patch_dim = patch_size
        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                     vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.post_init()

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

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, inputs_embeds=None, image_ids=None,
                head_mask=None, cross_attn_head_mask=None, past_key_values=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        hidden_states = self.dropout(inputs_embeds)

        position_bias = None
        for i, layer_module in enumerate(self.block):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                layer_head_mask=head_mask[i],
                use_cache=False,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if image_ids is not None:
            if image_ids.dim() > 3:
                image_ids = image_ids.view(image_ids.size(0), -1, image_ids.size(-1))
            image_embedding = self.image_dense(image_ids)
            if image_embedding.dim() > 3:
                image_embedding = image_embedding.view(image_embedding.size(0), -1, image_embedding.size(-1))
            image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)
            merge = torch.cat([hidden_states, image_att], dim=-1)
            gate = self.sigmoid(self.gate_dense(merge))
            hidden_states = (1 - gate) * hidden_states + gate * image_att

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)

class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    def __init__(self, config, patch_size=(100, 256)):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = JointEncoder(encoder_config, patch_size=patch_size)
        self.encoder.embed_tokens = self.shared
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)
        self.decoder.embed_tokens = self.shared
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, image_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, encoder_outputs=None, past_key_values=None,
                labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_ids=image_ids,
            )

        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
        )

        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                     encoder_outputs=None, **kwargs):
        image_ids = kwargs.get("image_ids", None)
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "image_ids": image_ids,
        }

# 3. Execute Stage 2 Training
base_model_name = 'allenai/unifiedqa-t5-base'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

ds_name = 'rad'
method = 'Second-Stage_Reasoning'
num_epochs = 20
learning_rate = 5e-5

print(f"============================================================")
print(f"🚀 Running Stage 2 Training ({method}) | Epochs: {num_epochs} | LR: {learning_rate}")
print(f"============================================================")

ckpt_out_dir = f'{CKPT_DIR}/{ds_name}_{method}'
os.makedirs(ckpt_out_dir, exist_ok=True)

model_obj = T5ForMultimodalGeneration.from_pretrained(
    base_model_name,
    patch_size=(100, 256)
)

train_dataset = OpenMedVQADataset(
    _tokenizer=tokenizer,
    _text_file_path=aug_train_json,
    _img_file_path=detr_pth,
    _img_name_map=name_map,
    _method=method,
    _dataset=ds_name,
    _source_len=512,
    _target_len=32
)

print(f"Train dataset size for Stage 2: {len(train_dataset)} samples")

training_args = Seq2SeqTrainingArguments(
    output_dir=ckpt_out_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=learning_rate,
    weight_decay=0.01,
    save_strategy='epoch',
    save_total_limit=1,
    eval_strategy='no',
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    report_to='none',
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    model=model_obj,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

print(f"🚀 Starting Stage 2 Training ({num_epochs} Epochs, LR={learning_rate})...")
train_result = trainer.train()
trainer.save_model(ckpt_out_dir)
print(f"✅ Stage 2 Training complete! Saved checkpoint to {ckpt_out_dir}")
