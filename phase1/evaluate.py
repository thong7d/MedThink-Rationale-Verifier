"""
evaluate.py — Inference & Evaluation script for MedThink.
Generates predictions, parses rationale & answer, computes accuracy, and saves predictions CSV.
"""

import os
import argparse
import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import setup_logging, get_device, extract_rationale_and_answer
from model import T5ForMultimodalGeneration
from dataset import MedVQADataset

def evaluate_medthink(args):
    setup_logging()
    logger = logging.getLogger(__name__)
    device = get_device()
    logger.info(f"Using device: {device}")

    # 1. Load Tokenizer & Model
    logger.info(f"Loading checkpoint from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = T5ForMultimodalGeneration.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # 2. Load Test Dataset
    logger.info(f"Loading test set from {args.test_json}...")
    test_dataset = MedVQADataset(
        tokenizer=tokenizer,
        data_json_path=args.test_json,
        feature_pth_path=args.feature_pth,
        name_map_json_path=args.name_map_json,
        method=args.strategy,
        task_type=args.task_type,
        dataset_name=args.dataset_name,
        source_len=args.max_source_len,
        target_len=args.max_target_len
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 3. Inference Loop
    results = []
    logger.info(f"Running inference ({len(test_dataset)} samples, strategy={args.strategy}, dataset={args.dataset_name})...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_ids = batch["image_ids"].to(device)

            # Generate outputs using beam search or greedy
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_ids=image_ids,
                max_length=args.max_gen_len,
                num_beams=args.num_beams,
                early_stopping=True
            )

            raw_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for i in range(len(raw_preds)):
                prob_id = batch["problem_id"][i]
                q_text = batch["question"][i]
                ans_gt = batch["answer_gt"][i]
                sol_gt = batch["solution_gt"][i]
                prompt = batch["prompt"][i]
                raw_out = raw_preds[i]

                # Extract rationale and answer from generation
                r_pred, a_pred = extract_rationale_and_answer(raw_out, method=args.strategy)

                # Determine correctness
                # For closed-end choice comparison:
                clean_a_pred = str(a_pred).strip().lower()
                clean_a_gt = str(ans_gt).strip().lower()

                # If answer choice extraction succeeded (e.g., 'A', 'B')
                if len(clean_a_pred) == 1 and clean_a_pred.isalpha():
                    # Format as (A) if gt is (A) or option letter
                    is_corr = (clean_a_pred in clean_a_gt)
                else:
                    is_corr = (clean_a_pred == clean_a_gt or clean_a_pred in clean_a_gt or clean_a_gt in clean_a_pred)

                results.append({
                    "problem_id": prob_id,
                    "dataset": args.dataset_name,
                    "strategy": args.strategy,
                    "question": q_text,
                    "answer_gt": ans_gt,
                    "answer_pred": a_pred,
                    "rationale_gt": sol_gt,
                    "rationale_pred": r_pred,
                    "is_correct": int(is_corr),
                    "prompt": prompt,
                    "raw_output": raw_out
                })

    df = pd.DataFrame(results)
    
    # 4. Compute Metrics
    acc = df["is_correct"].mean() * 100.0
    total = len(df)
    correct = df["is_correct"].sum()
    wrong = total - correct

    logger.info("=" * 60)
    logger.info(f"EVALUATION RESULTS — {args.dataset_name.upper()} ({args.strategy})")
    logger.info(f"Total Samples: {total}")
    logger.info(f"Correct: {correct} | Wrong: {wrong}")
    logger.info(f"Accuracy: {acc:.2f}%")
    logger.info("=" * 60)

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedThink Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test set JSON")
    parser.add_argument("--feature_pth", type=str, required=True, help="Path to DETR feature tensor")
    parser.add_argument("--name_map_json", type=str, required=True, help="Path to DETR name_map.json")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output predictions CSV")
    parser.add_argument("--strategy", type=str, default="Reasoning", choices=["Reasoning", "Explanation", "without_R"])
    parser.add_argument("--task_type", type=str, default="closed", choices=["closed", "open"])
    parser.add_argument("--dataset_name", type=str, default="rad")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    evaluate_medthink(args)
