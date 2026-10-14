"""
MedThink Phase 1: Faithful Baseline Reproduction for Google Colab
Target: R-RAD Closed-end (Explanation: ~83.5%, Reasoning: ~78.4%)
"""
import os, sys, glob, json, re, argparse, logging
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DetrForObjectDetection,
    DetrImageProcessor
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("MedThinkColab")

# ── Visual Encoder & Model Architecture ─────────────────────────────────────────
class VisualEncoder(nn.Module):
    def __init__(self, visual_dim=256, text_dim=768):
        super().__init__()
        self.proj = nn.Linear(visual_dim, text_dim)
        self.layer_norm = nn.LayerNorm(text_dim)

    def forward(self, visual_feats):
        projected = self.proj(visual_feats)
        return self.layer_norm(projected)

class Medthink_T5(nn.Module):
    def __init__(self, model_name="allenai/unifiedqa-t5-base", visual_dim=256):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        text_dim = self.t5.config.d_model
        self.visual_encoder = VisualEncoder(visual_dim=visual_dim, text_dim=text_dim)

    def forward(self, input_ids=None, attention_mask=None, image_ids=None, labels=None, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device
        text_embeds = self.t5.shared(input_ids)

        if image_ids is not None:
            img_embeds = self.visual_encoder(image_ids)
            img_len = img_embeds.size(1)
            img_mask = torch.ones((batch_size, img_len), dtype=attention_mask.dtype, device=device)
            inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
            extended_attention_mask = torch.cat([img_mask, attention_mask], dim=1)
        else:
            inputs_embeds = text_embeds
            extended_attention_mask = attention_mask

        return self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, input_ids=None, attention_mask=None, image_ids=None, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device
        text_embeds = self.t5.shared(input_ids)

        if image_ids is not None:
            img_embeds = self.visual_encoder(image_ids)
            img_len = img_embeds.size(1)
            img_mask = torch.ones((batch_size, img_len), dtype=attention_mask.dtype, device=device)
            inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
            extended_attention_mask = torch.cat([img_mask, attention_mask], dim=1)
        else:
            inputs_embeds = text_embeds
            extended_attention_mask = attention_mask

        return self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            **kwargs
        )

# ── Fast Dataset with Pre-tokenization & -100 Padding Mask ───────────────────────
class FastClosedMedVQADataset(Dataset):
    def __init__(self, data_list, tokenizer, visual_feats, name_map, method="Explanation",
                 max_source_len=512, max_target_len=256):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.visual_feats = visual_feats
        self.name_map = name_map
        self.method = method
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        logger.info(f"Pre-tokenizing {len(data_list)} samples for strategy '{method}'...")
        self.samples = []
        for item in data_list:
            q = item.get("Question", "").strip()
            options = item.get("Options", [])
            opt_str = " ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])

            if self.method == "Explanation":
                src_text = f"question: {q} options: {opt_str} context: visual features"
                rationale = item.get("Explanation", "").strip()
                ans = item.get("Answer", "").strip()
                tgt_text = f"explanation: {rationale} answer: {ans}"
            elif self.method == "Reasoning":
                src_text = f"question: {q} options: {opt_str} context: visual features"
                rationale = item.get("Reasoning", "").strip()
                ans = item.get("Answer", "").strip()
                tgt_text = f"reasoning: {rationale} answer: {ans}"
            else:
                src_text = f"question: {q} options: {opt_str} context: visual features"
                tgt_text = item.get("Answer", "").strip()

            img_name = item.get("Image_name", "")
            if img_name in self.name_map:
                feat_idx = self.name_map[img_name]
                img_feat = self.visual_feats[feat_idx]
            else:
                img_feat = torch.zeros((100, 256), dtype=torch.float32)

            src_enc = self.tokenizer(
                src_text,
                max_length=self.max_source_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            tgt_enc = self.tokenizer(
                tgt_text,
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Mask pad tokens with -100 so CrossEntropy ignores them
            labels = tgt_enc.input_ids.squeeze(0).clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            self.samples.append({
                "input_ids": src_enc.input_ids.squeeze(0),
                "attention_mask": src_enc.attention_mask.squeeze(0),
                "labels": labels,
                "image_ids": img_feat,
                "question": q,
                "answer_gt": item.get("Answer", "").strip(),
                "options": options,
            })
        logger.info(f"Done pre-tokenization for {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ── Evaluation & Parsing Utilities ──────────────────────────────────────────────
def extract_answer_from_generated(text, options=None):
    text = text.strip()
    match = re.search(r'answer:\s*(.*)', text, re.IGNORECASE)
    ans_candidate = match.group(1).strip() if match else text
    return ans_candidate.strip().rstrip(".").strip()

def is_prediction_correct(pred_ans, gt_ans):
    pred = pred_ans.strip().lower()
    gt = gt_ans.strip().lower()
    if pred == gt:
        return True
    if gt in pred or pred in gt:
        return True
    return False

class MultimodalSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_ids=inputs.get("image_ids", None),
            labels=inputs["labels"]
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if prediction_loss_only:
            with torch.no_grad():
                loss = self.compute_loss(model, inputs)
            return (loss, None, None)

        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_ids=inputs.get("image_ids", None),
                max_length=self.args.generation_max_length,
                num_beams=1,
                do_sample=False
            )
        labels = inputs.get("labels", None)
        return (None, generated_tokens, labels)

# ── DETR Extraction Helper ──────────────────────────────────────────────────────
def extract_or_load_detr(feature_dir, img_dir, device):
    detr_save_path = os.path.join(feature_dir, "rad_detr.pth")
    name_map_save_path = os.path.join(feature_dir, "name_map.json")

    # Check candidates from uploaded kaggle_outputs_v5 on Drive
    import shutil
    drive_root = os.path.dirname(feature_dir)
    candidates_detr = [
        os.path.join(drive_root, "kaggle_outputs_v5", "phase1_outputs", "detr_features", "R-RAD", "rad_detr.pth"),
        os.path.join(drive_root, "kaggle_outputs_v5", "detr_features", "R-RAD", "rad_detr.pth"),
        os.path.join(drive_root, "phase1_outputs", "detr_features", "R-RAD", "rad_detr.pth"),
        os.path.join(feature_dir, "rad_detr.pth")
    ]
    candidates_map = [
        os.path.join(drive_root, "kaggle_outputs_v5", "phase1_outputs", "detr_features", "R-RAD", "name_map.json"),
        os.path.join(drive_root, "kaggle_outputs_v5", "detr_features", "R-RAD", "name_map.json"),
        os.path.join(drive_root, "phase1_outputs", "detr_features", "R-RAD", "name_map.json"),
        os.path.join(feature_dir, "name_map.json")
    ]
    for c_pth, c_map in zip(candidates_detr, candidates_map):
        if os.path.exists(c_pth) and os.path.exists(c_map) and c_pth != detr_save_path:
            logger.info(f"Found existing DETR features from kaggle_outputs_v5: {c_pth}")
            os.makedirs(feature_dir, exist_ok=True)
            shutil.copy2(c_pth, detr_save_path)
            shutil.copy2(c_map, name_map_save_path)
            break

    if os.path.exists(detr_save_path) and os.path.exists(name_map_save_path):
        logger.info(f"Loaded existing DETR features from {detr_save_path}")
        feats = torch.load(detr_save_path, map_location="cpu")
        with open(name_map_save_path, "r", encoding="utf-8") as f:
            name_map = json.load(f)
        return feats, name_map

    logger.info("Extracting DETR features from raw images...")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    assert len(img_files) > 0, f"No images found in {img_dir}"

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    detr_model.eval()

    all_features = []
    name_map = {}
    batch_size = 16

    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size), desc="DETR Extraction"):
            batch_files = img_files[i:i + batch_size]
            batch_imgs = []
            for f in batch_files:
                try:
                    batch_imgs.append(Image.open(f).convert("RGB"))
                except Exception:
                    batch_imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            outputs = detr_model(**inputs, output_hidden_states=True)
            feat = outputs.last_hidden_state.cpu()

            start_idx = len(all_features)
            for j, f in enumerate(batch_files):
                img_name = os.path.basename(f)
                name_map[img_name] = start_idx + j
                all_features.append(feat[j])

    feats = torch.stack(all_features, dim=0)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feats, detr_save_path)
    with open(name_map_save_path, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2)
    logger.info(f"Saved DETR features {feats.shape} to {detr_save_path}")
    return feats, name_map

# ── Experiment Runner ────────────────────────────────────────────────────────────
def train_and_eval_strategy(strategy, train_data, test_data, tokenizer, visual_feats, name_map, output_dir, device, epochs=150):
    logger.info("=" * 60)
    logger.info(f"STARTING STRATEGY: {strategy} (150 epochs)")
    logger.info("=" * 60)

    train_ds = FastClosedMedVQADataset(train_data, tokenizer, visual_feats, name_map, method=strategy)
    test_ds = FastClosedMedVQADataset(test_data, tokenizer, visual_feats, name_map, method=strategy)

    model = Medthink_T5("allenai/unifiedqa-t5-base", visual_dim=256).to(device)
    model_output_dir = os.path.join(output_dir, f"model_R-RAD_{strategy}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=256,
        fp16=False,
        report_to=["none"],
        remove_unused_columns=False,
        dataloader_num_workers=2
    )

    trainer = MultimodalSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer
    )

    trainer.train()

    logger.info(f"Running inference on test set ({len(test_ds)} samples)...")
    predict_res = trainer.predict(test_ds)
    preds = predict_res.predictions
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    results = []
    correct = 0
    for idx, (p_text, sample) in enumerate(zip(decoded_preds, test_ds.samples)):
        ans_pred = extract_answer_from_generated(p_text, sample["options"])
        is_corr = is_prediction_correct(ans_pred, sample["answer_gt"])
        if is_corr:
            correct += 1
        results.append({
            "question": sample["question"],
            "raw_output": p_text,
            "answer_pred": ans_pred,
            "answer_gt": sample["answer_gt"],
            "is_correct": is_corr
        })

    acc = correct / len(results) * 100.0
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"predictions_R-RAD_{strategy.lower()}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info(f"RESULTS R-RAD {strategy}: Accuracy = {acc:.2f}% ({correct}/{len(results)})")
    logger.info(f"Saved predictions to: {csv_path}")
    logger.info("=" * 60)
    return acc, df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, default="/content/Medthink_repo")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/MedThink_Experiments/phase1_outputs")
    parser.add_argument("--feature_dir", type=str, default="/content/drive/MyDrive/MedThink_Experiments/detr_features")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--strategy", type=str, choices=["Explanation", "Reasoning", "both"], default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hardware Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feature_dir, exist_ok=True)

    dataset_root = os.path.join(args.repo_dir, "Medthink_Dataset", "R-RAD")
    train_json_path = os.path.join(dataset_root, "closed_end", "train.json")
    test_json_path = os.path.join(dataset_root, "closed_end", "test.json")
    img_dir = os.path.join(dataset_root, "images")

    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    visual_feats, name_map = extract_or_load_detr(args.feature_dir, img_dir, device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-base")

    summary = []
    if args.strategy in ["Explanation", "both"]:
        acc_exp, _ = train_and_eval_strategy("Explanation", train_data, test_data, tokenizer, visual_feats, name_map, args.output_dir, device, args.epochs)
        summary.append({"Strategy": "Explanation", "Paper": "83.5%", "Colab": f"{acc_exp:.2f}%"})

    if args.strategy in ["Reasoning", "both"]:
        acc_rea, df_rea = train_and_eval_strategy("Reasoning", train_data, test_data, tokenizer, visual_feats, name_map, args.output_dir, device, args.epochs)
        summary.append({"Strategy": "Reasoning", "Paper": "78.4%", "Colab": f"{acc_rea:.2f}%"})

        # Bridge to Phase 2
        df_weak = df_rea.copy()
        df_weak["label"] = df_weak["is_correct"].apply(lambda x: 1 if x else 0)
        weak_csv = os.path.join(os.path.dirname(args.output_dir), "verifier_data", "weak_labels_R-RAD_reasoning.csv")
        os.makedirs(os.path.dirname(weak_csv), exist_ok=True)
        df_weak.to_csv(weak_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Saved Phase 2 weak labels to {weak_csv}")

    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(args.output_dir, "baseline_accuracy_summary.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("Summary of Reproduction:")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
