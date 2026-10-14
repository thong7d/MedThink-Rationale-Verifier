"""
train.py — Training script for MedThink (T5-base + DETR Gated Fusion).
Supports training on R-RAD, R-SLAKE, and R-PathVQA datasets with Reasoning or Explanation strategies.
"""

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils import setup_logging, set_seed, get_device
from model import T5ForMultimodalGeneration
from dataset import MedVQADataset

def train_medthink(args):
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # 1. Load Tokenizer & Model
    model_name = args.base_model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForMultimodalGeneration.from_pretrained(model_name, attn_implementation="eager")
    model.to(device)

    # 2. Load Datasets
    logger.info(f"Loading training data from {args.train_json}...")
    train_dataset = MedVQADataset(
        tokenizer=tokenizer,
        data_json_path=args.train_json,
        feature_pth_path=args.feature_pth,
        name_map_json_path=args.name_map_json,
        method=args.strategy,
        task_type=args.task_type,
        dataset_name=args.dataset_name,
        source_len=args.max_source_len,
        target_len=args.max_target_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    eval_loader = None
    if args.val_json and os.path.exists(args.val_json):
        val_dataset = MedVQADataset(
            tokenizer=tokenizer,
            data_json_path=args.val_json,
            feature_pth_path=args.feature_pth,
            name_map_json_path=args.name_map_json,
            method=args.strategy,
            task_type=args.task_type,
            dataset_name=args.dataset_name,
            source_len=args.max_source_len,
            target_len=args.max_target_len
        )
        eval_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )

    # 4. Training Loop
    logger.info(f"Starting Training ({args.epochs} epochs, strategy={args.strategy}, dataset={args.dataset_name})...")
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_ids = batch["image_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_ids=image_ids,
                labels=labels
            )
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            total_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum_steps:.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} complete — Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info(f"Saved best model checkpoint to {best_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training completed. Final checkpoint saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedThink Model")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, default="")
    parser.add_argument("--feature_pth", type=str, required=True)
    parser.add_argument("--name_map_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="Reasoning", choices=["Reasoning", "Explanation", "without_R"])
    parser.add_argument("--task_type", type=str, default="closed", choices=["closed", "open"])
    parser.add_argument("--dataset_name", type=str, default="rad")
    parser.add_argument("--base_model", type=str, default="allenai/unifiedqa-t5-base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_medthink(args)
