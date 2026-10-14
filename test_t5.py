import torch
from transformers import T5Config, AutoTokenizer
import sys
import copy
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

sys.path.append(r"d:\000MINHTHONG\Graduation Thesis\experiments\medThink\kaggle_push")
from medthink_phase1_kaggle import T5ForMultimodalGeneration

def main():
    model_name = "allenai/unifiedqa-t5-base"
    print("Loading model...")
    model = T5ForMultimodalGeneration.from_pretrained(model_name, patch_size=(100, 256))
    
    print("Creating dummy inputs...")
    batch_size = 4
    enc_seq_len = 1024
    dec_seq_len = 512
    
    input_ids = torch.randint(0, 1000, (batch_size, enc_seq_len))
    attention_mask = torch.ones(batch_size, enc_seq_len)
    labels = torch.randint(0, 1000, (batch_size, dec_seq_len))
    image_ids = torch.randn(batch_size, 100, 256)
    
    print("Running forward pass...")
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_ids=image_ids,
            labels=labels
        )
        print("Forward pass successful!")
    except Exception as e:
        print("Error during forward pass:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
