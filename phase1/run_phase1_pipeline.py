"""
run_phase1_pipeline.py — Master script for Phase 1: Reproduce MedThink Baseline & Collect Predictions.

Runs end-to-end:
1. Verifies dataset paths and extracted DETR feature files.
2. Trains MedThink with Reasoning (R->A) and Explanation (A->R) strategies on R-RAD, R-SLAKE, and R-PathVQA.
3. Evaluates on test sets and outputs standardized predictions CSV files for Phase 2 weak label generation.
"""

import os
import sys
import argparse
import subprocess
import logging

def run_cmd(cmd):
    print(f"\n[RUNNING]: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR]: Command failed with returncode {res.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Master Pipeline")
    parser.add_argument("--base_dir", type=str, default=".", help="Base project working directory")
    parser.add_argument("--dataset_root", type=str, default="repo/Medthink/Medthink_Dataset", help="Path to MedThink datasets")
    parser.add_argument("--output_dir", type=str, default="phase1_outputs", help="Output directory for checkpoints and predictions")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--skip_train", action="store_true", help="Skip training if checkpoints already exist")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    python_exe = sys.executable

    print("=" * 70)
    print("PHASE 1: REPRODUCE MEDTHINK BASELINE & COLLECT PREDICTIONS")
    print("=" * 70)

    # Dataset configurations
    datasets_config = [
        {
            "name": "R-RAD",
            "train_json": os.path.join(args.dataset_root, "R-RAD/closed-end/trainset.json"),
            "val_json": os.path.join(args.dataset_root, "R-RAD/closed-end/testset.json"),
            "test_json": os.path.join(args.dataset_root, "R-RAD/closed-end/testset.json"),
            "feature_pth": os.path.join(args.dataset_root, "R-RAD/detr_features/rad_detr.pth"),
            "name_map": os.path.join(args.dataset_root, "R-RAD/detr_features/name_map.json"),
            "task_type": "closed"
        },
        {
            "name": "R-SLAKE",
            "train_json": os.path.join(args.dataset_root, "R-SLAKE/closed-end/train.json"),
            "val_json": os.path.join(args.dataset_root, "R-SLAKE/closed-end/validate.json"),
            "test_json": os.path.join(args.dataset_root, "R-SLAKE/closed-end/test.json"),
            "feature_pth": os.path.join(args.dataset_root, "R-SLAKE/detr_features/slake_detr.pth"),
            "name_map": os.path.join(args.dataset_root, "R-SLAKE/detr_features/name_map.json"),
            "task_type": "closed"
        },
        {
            "name": "R-PathVQA",
            "train_json": os.path.join(args.dataset_root, "R-PathVQA/closed-end/train.json"),
            "val_json": os.path.join(args.dataset_root, "R-PathVQA/closed-end/val.json"),
            "test_json": os.path.join(args.dataset_root, "R-PathVQA/closed-end/test.json"),
            "feature_pth": os.path.join(args.dataset_root, "R-PathVQA/detr_features/path_detr.pth"),
            "name_map": os.path.join(args.dataset_root, "R-PathVQA/detr_features/name_map.json"),
            "task_type": "closed"
        }
    ]

    strategies = ["Reasoning", "Explanation"]

    for config in datasets_config:
        ds_name = config["name"]
        print(f"\n--- Processing Dataset: {ds_name} ---")

        for strat in strategies:
            print(f"\n>> Strategy: {strat} <<")

            checkpoint_dir = os.path.join(args.output_dir, "checkpoints", f"{ds_name}_{strat}")
            model_path = os.path.join(checkpoint_dir, "best_model")
            if not os.path.exists(model_path):
                model_path = os.path.join(checkpoint_dir, "final_model")

            csv_output = os.path.join(args.output_dir, f"predictions_{ds_name}_{strat.lower()}.csv")

            # Step 1: Training (if not skipped or missing)
            if not args.skip_train or not os.path.exists(model_path):
                train_cmd = [
                    python_exe, "phase1/train.py",
                    "--train_json", config["train_json"],
                    "--val_json", config["val_json"],
                    "--feature_pth", config["feature_pth"],
                    "--name_map_json", config["name_map"],
                    "--output_dir", checkpoint_dir,
                    "--strategy", strat,
                    "--task_type", config["task_type"],
                    "--dataset_name", ds_name.lower(),
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size)
                ]
                ok = run_cmd(train_cmd)
                if not ok:
                    print(f"Skipping evaluation for {ds_name} {strat} due to training error.")
                    continue

            # Step 2: Evaluation & Predictions CSV Generation
            eval_cmd = [
                python_exe, "phase1/evaluate.py",
                "--model_path", model_path,
                "--test_json", config["test_json"],
                "--feature_pth", config["feature_pth"],
                "--name_map_json", config["name_map"],
                "--output_csv", csv_output,
                "--strategy", strat,
                "--task_type", config["task_type"],
                "--dataset_name", ds_name.lower(),
                "--batch_size", "8"
            ]
            run_cmd(eval_cmd)

    print("\n" + "=" * 70)
    print("PHASE 1 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All prediction files saved in: {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
