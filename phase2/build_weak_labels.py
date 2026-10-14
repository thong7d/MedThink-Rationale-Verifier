"""
build_weak_labels.py — Phase 2: Weak Label Construction & Stratified Data Splitting.

Reads Phase 1 prediction CSV files, constructs binary weak labels (1=Good, 0=Bad rationale),
and performs stratified train/val/test splits for training the Verifier.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def process_predictions_csv(csv_path, output_dir, dataset_name="rad", strategy="reasoning", seed=42):
    """
    Process prediction CSV:
    1. Validate required columns.
    2. Add `weak_label` = `is_correct`.
    3. Perform stratified 60/20/20 train/val/test split.
    4. Save split CSVs to `output_dir`.
    """
    if not os.path.exists(csv_path):
        logging.error(f"Input file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} predictions from {csv_path}")

    # Ensure required columns exist
    if "is_correct" not in df.columns:
        raise ValueError("Input CSV missing 'is_correct' column!")

    df["weak_label"] = df["is_correct"].astype(int)

    n_total = len(df)
    n_pos = df["weak_label"].sum()
    n_neg = n_total - n_pos
    pos_rate = (n_pos / n_total) * 100.0 if n_total > 0 else 0.0

    logging.info("=" * 60)
    logging.info(f"DATASET: {dataset_name.upper()} | STRATEGY: {strategy.upper()}")
    logging.info(f"Total Samples: {n_total}")
    logging.info(f"Good Rationales (Label 1): {n_pos} ({pos_rate:.1f}%)")
    logging.info(f"Bad Rationales  (Label 0): {n_neg} ({100.0 - pos_rate:.1f}%)")
    logging.info("=" * 60)

    # Perform stratified split: 60% train, 20% val, 20% test
    # Handles edge case when class count is too small for stratification
    min_class_count = df["weak_label"].value_counts().min()
    use_stratify = min_class_count >= 3

    if use_stratify:
        train_df, temp_df = train_test_split(
            df, test_size=0.40, stratify=df["weak_label"], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["weak_label"], random_state=seed
        )
    else:
        logging.warning("Class count too small for stratification, falling back to random split.")
        train_df, temp_df = train_test_split(df, test_size=0.40, random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)

    # Save outputs
    verifier_dir = os.path.join(output_dir, "verifier_data")
    os.makedirs(verifier_dir, exist_ok=True)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    summary_rows = []
    for split_name, split_df in splits.items():
        out_file = os.path.join(verifier_dir, f"{dataset_name.lower()}_{strategy.lower()}_{split_name}.csv")
        split_df.to_csv(out_file, index=False, encoding="utf-8-sig")

        s_pos = split_df["weak_label"].sum()
        s_total = len(split_df)
        s_rate = (s_pos / s_total * 100.0) if s_total > 0 else 0.0

        logging.info(f"Saved {split_name.upper()} split -> {out_file} ({s_total} samples, {s_rate:.1f}% positive)")

        summary_rows.append({
            "dataset": dataset_name,
            "strategy": strategy,
            "split": split_name,
            "total_samples": s_total,
            "positive_samples": s_pos,
            "negative_samples": s_total - s_pos,
            "positive_rate_pct": round(s_rate, 2)
        })

    return pd.DataFrame(summary_rows)

def main():
    parser = argparse.ArgumentParser(description="Build Weak Labels & Stratified Splits")
    parser.add_argument("--predictions_csv", type=str, required=True, help="Input predictions CSV from Phase 1")
    parser.add_argument("--output_dir", type=str, default="phase2_outputs", help="Output directory for verifier data")
    parser.add_argument("--dataset_name", type=str, default="rad", help="Dataset name")
    parser.add_argument("--strategy", type=str, default="reasoning", help="Strategy name (reasoning or explanation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    summary_df = process_predictions_csv(
        csv_path=args.predictions_csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        strategy=args.strategy,
        seed=args.seed
    )

    if summary_df is not None:
        stats_path = os.path.join(args.output_dir, f"weak_label_stats_{args.dataset_name}_{args.strategy}.csv")
        summary_df.to_csv(stats_path, index=False)
        print(f"Summary saved to {stats_path}")

if __name__ == "__main__":
    main()
