"""
dataset.py — PyTorch Dataset classes for MedThink VQA.
Supports R-RAD, R-SLAKE, and R-PathVQA datasets (Closed-end & Open-end).
Formats inputs and targets for Reasoning (R->A) and Explanation (A->R) strategies.
"""

import os
import json
import torch
from torch.utils.data import Dataset

class MedVQADataset(Dataset):
    """
    Unified Dataset class for MedThink Visual Question Answering.
    
    Supports:
    - Task Types: 'closed' (multiple choice) or 'open' (open-ended answer)
    - Methods: 'Reasoning' (R->A), 'Explanation' (A->R), 'without_R' (A only), etc.
    - Features: DETR visual features tensor [num_images, 100, 256] + name_map JSON
    """
    
    OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F']

    def __init__(self, tokenizer, data_json_path, feature_pth_path, name_map_json_path,
                 method="Reasoning", task_type="closed", dataset_name="rad",
                 source_len=512, target_len=512):
        self.tokenizer = tokenizer
        self.method = method
        self.task_type = task_type
        self.dataset_name = dataset_name.lower()
        self.source_len = source_len
        self.target_len = target_len

        # Load DETR features tensor and name mapping
        if isinstance(feature_pth_path, torch.Tensor):
            self.pretrained_feature = feature_pth_path
        else:
            self.pretrained_feature = torch.load(feature_pth_path, map_location="cpu")
            
        with open(name_map_json_path, "r", encoding="utf-8") as f:
            self.name_map = json.load(f)

        # Load QA JSON data
        with open(data_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples = []
        for pid, problem in raw_data.items():
            parsed = self._parse_problem(pid, problem)
            if parsed is not None:
                self.samples.append(parsed)

    def _parse_problem(self, pid, problem):
        question = problem.get("question", "").strip()
        solution = problem.get("solution", "").strip()
        choices = problem.get("choices", [])
        answer_idx = problem.get("answer", 0)

        # Extract image lookup key
        if "image" in problem:
            img_key = problem["image"].replace(".jpg", "").replace(".png", "")
        elif "img_id" in problem:
            img_key = str(problem["img_id"])
        elif "img_name" in problem:
            img_key = str(problem["img_name"]).replace(".jpg", "").replace(".png", "")
        else:
            img_key = str(pid)

        if img_key not in self.name_map:
            # Fallback search if exact key match fails
            matched_key = None
            for k in self.name_map.keys():
                if k in img_key or img_key in k:
                    matched_key = k
                    break
            if matched_key:
                img_key = matched_key
            else:
                return None

        img_idx = int(self.name_map[img_key])

        # Construct prompt and target based on task_type and method
        if self.task_type == "closed":
            choice_str = " ".join([f"({self.OPTIONS[i]}) {c}" for i, c in enumerate(choices)])
            if isinstance(answer_idx, int) and answer_idx < len(choices):
                answer_text = f"({self.OPTIONS[answer_idx]})"
                gt_answer = choices[answer_idx]
            else:
                answer_text = f"({self.OPTIONS[0]})"
                gt_answer = str(answer_idx)

            if self.method == "Reasoning":
                prompt = f"Question: {question}\nOptions: {choice_str}\nSolution:"
                target = f"{solution}\nAnswer: The answer is {answer_text}."
            elif self.method == "Explanation":
                prompt = f"Question: {question}\nOptions: {choice_str}\nAnswer:"
                target = f"The answer is {answer_text}.\nSolution: {solution}"
            elif self.method == "without_R":
                prompt = f"Question: {question}\nOptions: {choice_str}\nAnswer:"
                target = f"The answer is {answer_text}."
            else:
                prompt = f"Question: {question}\nOptions: {choice_str}\nSolution:"
                target = f"{solution}\nAnswer: The answer is {answer_text}."

        else:  # open-ended VQA
            gt_answer = choices[0] if choices else str(answer_idx)
            if self.method == "Reasoning":
                prompt = f"Question: {question}\nSolution:"
                target = f"{solution}\nAnswer: The answer is {gt_answer}."
            elif self.method == "Explanation":
                prompt = f"Question: {question}\nAnswer:"
                target = f"The answer is {gt_answer}.\nSolution: {solution}"
            elif self.method == "without_R":
                prompt = f"Question: {question}\nAnswer:"
                target = f"The answer is {gt_answer}."
            else:
                prompt = f"Question: {question}\nSolution:"
                target = f"{solution}\nAnswer: The answer is {gt_answer}."

        return {
            "problem_id": pid,
            "img_key": img_key,
            "img_idx": img_idx,
            "question": question,
            "solution_gt": solution,
            "answer_gt": gt_answer,
            "choices": choices,
            "prompt": " ".join(prompt.split()),
            "target": " ".join(target.split())
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        source_enc = self.tokenizer(
            sample["prompt"],
            max_length=self.source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            sample["target"],
            max_length=self.target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        img_idx = sample["img_idx"]
        image_ids = self.pretrained_feature[img_idx]

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "problem_id": sample["problem_id"],
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "image_ids": image_ids,
            "labels": labels,
            "prompt": sample["prompt"],
            "target": sample["target"],
            "question": sample["question"],
            "solution_gt": sample["solution_gt"],
            "answer_gt": sample["answer_gt"]
        }
