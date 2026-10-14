"""
utils.py — Shared utility functions for the MedThink Verifier project.
"""

import os
import re
import json
import logging
import numpy as np
import torch


def setup_logging(level=logging.INFO):
    """Configure logging for the project."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger("medthink")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def extract_answer_closed(text: str) -> str:
    """
    Extract the answer choice (A/B) from model output text.
    Handles both Reasoning and Explanation output formats.
    """
    pattern = re.compile(r"The answer is \(([A-Z])\)")
    matches = pattern.findall(text)
    if len(matches) >= 1:
        return matches[-1]  # Take the last match (answer comes last in Reasoning)
    return "FAILED"


def extract_rationale_and_answer(text: str, method: str):
    """
    Parse model output into (rationale, answer) parts.
    
    For Reasoning strategy output format:
        "{rationale}\nAnswer: The answer is (X)."
    
    For Explanation strategy output format:
        "The answer is (X).\nSolution: {rationale}"
    """
    text = text.strip()

    if method == "Reasoning":
        # Format: "{rationale}\nAnswer: The answer is (X)."
        parts = text.rsplit("\nAnswer:", 1)
        if len(parts) == 2:
            rationale = parts[0].strip()
            answer_text = parts[1].strip()
        else:
            rationale = text
            answer_text = text
        answer = extract_answer_closed(answer_text)

    elif method == "Explanation":
        # Format: "The answer is (X).\nSolution: {rationale}"
        parts = text.split("\nSolution:", 1)
        if len(parts) == 2:
            answer_text = parts[0].strip()
            rationale = parts[1].strip()
        else:
            answer_text = text
            rationale = text
        answer = extract_answer_closed(answer_text)

    else:
        rationale = text
        answer = extract_answer_closed(text)

    return rationale, answer


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)
