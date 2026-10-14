"""
extract_detr_features.py — Extract DETR visual feature embeddings for MedThink VQA.
Saves pre-extracted feature tensors [num_images, 100, 256] and name_map.json.
"""

import os
import json
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

def get_detr_model(device="cuda"):
    """
    Loads DETR ResNet-101 model for feature extraction.
    Tries torch.hub first, falls back to HuggingFace transformers.
    """
    print("Loading DETR ResNet-101 feature extractor...")
    try:
        model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101', pretrained=True)
        model.to(device)
        model.eval()
        
        # Strip classification/bbox heads to keep sequence of object queries (100, 256)
        class DetrFeatureExtractor(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                # Standard DETR forward return intermediate decoder states or object query features
                out = self.base_model.backbone(x)
                src, mask = out[0].decompose()
                hs = self.base_model.transformer(self.base_model.input_proj(src), mask, self.base_model.query_embed.weight, self.base_model.pos_embed)[0]
                return hs[-1] # Shape: [1, 100, 256]
                
        feature_extractor = DetrFeatureExtractor(model)
        print("Loaded DETR via torch.hub successfully.")
    except Exception as e:
        print(f"Torch hub DETR load failed ({e}), falling back to HuggingFace DetrModel...")
        from transformers import DetrModel
        model = DetrModel.from_pretrained("facebook/detr-resnet-101")
        model.to(device)
        model.eval()

        class HfDetrFeatureExtractor(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                outputs = self.base_model(pixel_values=x)
                return outputs.last_hidden_state # Shape: [1, 100, 256]

        feature_extractor = HfDetrFeatureExtractor(model)
        print("Loaded DETR via HuggingFace successfully.")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return feature_extractor, transform

def extract_dataset_features(image_dir, output_dir, dataset_name="rad", device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    model, transform = get_detr_model(device=device)

    image_files = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))

    image_files.sort()
    print(f"Extracting DETR features for {len(image_files)} images from {image_dir}...")

    name_map = {}
    vision_features = []

    for idx, img_path in enumerate(tqdm(image_files, desc="Extracting DETR Features")):
        try:
            img = Image.open(img_path).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(inp).squeeze(0).cpu() # [100, 256]

            # Key name extraction based on dataset
            img_basename = os.path.basename(img_path)
            img_key = os.path.splitext(img_basename)[0]
            
            name_map[img_key] = str(idx)
            vision_features.append(feat)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if len(vision_features) > 0:
        features_tensor = torch.stack(vision_features, dim=0) # [N, 100, 256]
        output_pth = os.path.join(output_dir, f"{dataset_name}_detr.pth")
        output_map = os.path.join(output_dir, "name_map.json")

        torch.save(features_tensor, output_pth)
        with open(output_map, "w", encoding="utf-8") as f:
            json.dump(name_map, f, indent=4, ensure_ascii=False)

        print(f"Features saved to {output_pth} (Shape: {features_tensor.shape})")
        print(f"Name map saved to {output_map} ({len(name_map)} entries)")
    else:
        print("No images found or extracted!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR Visual Feature Extractor")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to raw dataset images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted .pth and name_map.json")
    parser.add_argument("--dataset", type=str, default="rad", choices=["rad", "slake", "path"], help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    extract_dataset_features(args.image_dir, args.output_dir, dataset_name=args.dataset, device=args.device)
