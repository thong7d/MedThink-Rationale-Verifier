# Phase 2: Build Weak Labels for Rationale Verification

## Overview
Phase 2 transforms the Phase 1 prediction CSV outputs into a labeled dataset for training and evaluating the Rationale Verifier.

## Weak Label Definition
- **Label 1 (Good Rationale)**: `A_pred == A_gt` (Model generated correct answer, rationale is likely faithful and accurate).
- **Label 0 (Bad Rationale)**: `A_pred != A_gt` (Model generated incorrect answer, rationale contains hallucination or flawed reasoning).

## Data Splits
For each dataset and strategy:
- **Train Split (60%)**: Used to train the Verifier models (BiomedCLIP, DeBERTa, Ensemble).
- **Validation Split (20%)**: Used for hyperparameter tuning and learning adaptive verification thresholds.
- **Test Split (20%)**: Held out for final evaluation (ROC-AUC, Precision, Recall, Accuracy recovery).

## Key Output Files
- `phase2_outputs/verifier_data/{dataset}_{strategy}_train.csv`
- `phase2_outputs/verifier_data/{dataset}_{strategy}_val.csv`
- `phase2_outputs/verifier_data/{dataset}_{strategy}_test.csv`
- `phase2_outputs/weak_label_stats.csv`
