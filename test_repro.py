import torch
from transformers import AutoTokenizer
from kaggle_push.medthink_phase1_kaggle import T5ForMultimodalGeneration, MedVQADataset

model_name = "allenai/unifiedqa-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForMultimodalGeneration.from_pretrained(model_name, patch_size=(100, 256), attn_implementation="eager")
model.train()

# Mock tensors directly
batch_size = 4
input_ids = torch.randint(0, 1000, (batch_size, 512))
attention_mask = torch.ones(batch_size, 512)
attention_mask[:, 256:] = 0
labels = torch.randint(0, 1000, (batch_size, 512))
image = torch.randn(batch_size, 100, 256)




print(f"input_ids: {input_ids.shape}")
print(f"attention_mask: {attention_mask.shape}")
print(f"labels: {labels.shape}")
print(f"image: {image.shape}")

out = model(input_ids=input_ids, attention_mask=attention_mask, image_ids=image, labels=labels)
print(out.loss)
