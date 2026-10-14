import torch
from medthink_phase1_kaggle import T5ForMultimodalGeneration
from transformers import AutoTokenizer

model_name = "allenai/unifiedqa-t5-base"
model = T5ForMultimodalGeneration.from_pretrained(model_name, patch_size=(100, 256), attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = torch.randint(0, 1000, (4, 512))
attention_mask = torch.ones((4, 512), dtype=torch.long)
image_ids = torch.randn(4, 100, 256)
labels = torch.randint(0, 1000, (4, 512))

try:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_ids=image_ids,
        labels=labels
    )
    print("Forward pass successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
