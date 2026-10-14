# Phase 1: Reproduce MedThink Baseline & Collect Predictions

## Overview
This phase reproduces the MedThink baseline on all 3 datasets (R-RAD, R-SLAKE, R-PathVQA) 
with both **Reasoning** (R→A) and **Explanation** (A→R) strategies.

## Directory Structure
```
phase1/
├── README.md                    # This file
├── model.py                     # MedThink model (T5ForMultimodalGeneration)
├── dataset.py                   # Dataset classes (fixed bugs from original)
├── extract_features.py          # DETR feature extraction
├── train.py                     # Training script
├── inference.py                 # Inference & prediction collection
├── run_all.py                   # End-to-end pipeline (for Kaggle)
└── utils.py                     # Shared utilities
```

## Quick Start (Kaggle)
Upload all files + upload datasets to Kaggle dataset.
Run `run_all.py` which orchestrates:
1. DETR feature extraction from images
2. Training closed-end + open-end models with Reasoning strategy
3. Inference on test sets
4. Save predictions CSV (for Phase 2 verifier)

## Expected Results (from paper)
| Dataset  | Split  | Reasoning | Explanation |
|----------|--------|-----------|-------------|
| R-RAD    | closed | ~78.4%    | ~83.5%      |
| R-SLAKE  | closed | ~81.2%    | ~86.3%      |
| R-RAD    | open   | varies    | varies      |
| R-SLAKE  | open   | varies    | varies      |
