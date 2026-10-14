# Chẩn đoán & Khắc phục Sự Bất ổn của Chiến lược "Reasoning" bằng Cơ chế Tự-Xác Thực Rationale cho MedThink

## Tổng quan

**Paper gốc:** MedThink: A Rationale-Guided Framework for Explaining Medical Visual Question Answering (Gai et al., NAACL Findings 2025)

**Vấn đề cốt lõi từ paper:**
- MedThink đạt 83.5% (R-RAD), 86.3% (R-SLAKE), 87.2% (R-Path) với chiến lược **Explanation** (sinh A trước R)
- Chiến lược **Reasoning** (sinh R trước A) giảm accuracy ~5.1 điểm trên R-RAD — rationale sai lan truyền thành answer sai (hallucination, Bảng 4)
- Paper tự nhận hạn chế về "Trust and Reliability" nhưng không cung cấp cơ chế phát hiện khi nào rationale không đáng tin

**Đóng góp đề xuất:** Xây module verifier nhẹ chạy post-hoc, gắn **confidence score / uncertainty flag** cho mỗi cặp {R, A}, trả lời câu hỏi: *"khi nào tôi KHÔNG nên tin rationale này?"*

**Kiến trúc gốc MedThink (giữ nguyên):**
- Text encoder: UnifiedQA (T5-base, ~223M params)
- Visual encoder: DETR (Detection Transformer)
- Cross-attention + Gated Fusion → T5 decoder sinh R rồi A (Reasoning strategy)

---

## User Review Required

> [!IMPORTANT]
> **Lựa chọn GPU trên Colab:** Plan giả định bạn sử dụng Colab Pro/Pro+ với GPU T4 hoặc A100. Free tier (T4, ~15GB VRAM) đủ cho hầu hết thí nghiệm nhưng sẽ chậm hơn đáng kể ở Phase 5 (multi-sampling). Nếu dùng cloud GPU (Vast.ai/RunPod) thì ước tính 20-40 GPU-giờ ≈ $15-40.

> [!WARNING]
> **Rủi ro negative result:** Nếu verifier không phân biệt rationale tốt/xấu tốt hơn random, kết quả vẫn có giá trị khoa học (chứng minh rationale của MedThink nhất quán semantically ngay cả khi sai), nhưng cần chuẩn bị narrative phù hợp cho bảo vệ khóa luận.

> [!CAUTION]
> **Cỡ mẫu nhỏ ở một số vùng giải phẫu:** Pelvic Cavity N=22, Chest N=5 (Bảng 4) → bootstrap confidence interval thay vì p-value truyền thống. Cần nêu rõ limitation này.

## Quyết định Design (Đã xác nhận)

> [!IMPORTANT]
> **Q1 — Chiến lược target:** ✅ Chạy verifier trên **cả 2 chiến lược**:
> - **Reasoning (R→A)**: mục tiêu chính (nơi hallucination xảy ra)
> - **Explanation (A→R)**: chạy song song để so sánh — insight quan trọng: verifier có dễ hơn khi R được sinh SAU A không?
> 
> **Q2 — Threshold Strategy:** ✅ **Kết hợp Fixed + Adaptive**:
> 1. Học Adaptive threshold trên **R-SLAKE val** (dataset lớn nhất, đủ N/class)
> 2. Áp cố định threshold đó lên R-RAD và R-Path test (cross-dataset generalization)
> 3. Báo cáo cả 3 kịch bản: Adaptive per-dataset | Fixed 20th-pct | Cross-dataset
> 4. Với vùng N<30 (Pelvic, Chest): **luôn dùng cross-dataset threshold**, không learn trực tiếp
> 
> **Q3 — CLIP Model:** ✅ **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)
> - Trained trên 15M biomedical figure-caption pairs → domain-specific
> - PubMedBERT text encoder + ViT image encoder
> - API: `open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-...')`

---

## Proposed Changes — Kế hoạch Thực nghiệm 8 Giai đoạn

### Phase 0A: Download Dataset & EDA (Notebook 0 — **script riêng biệt**)

**File:** [`00_download_and_eda.py`](file:///d:/000MINHTHONG/Graduation%20Thesis/experiments/medThink/00_download_and_eda.py)

**Mục tiêu:** Kéo toàn bộ dataset R-RAD/R-SLAKE/R-Path về Colab Drive; EDA đầy đủ để hiểu phân phối trước khi chạy thí nghiệm.

**19 Cells bao gồm:**

| Cell | Nội dung |
|------|----------|
| 1 | Setup: Mount Drive, install packages |
| 2 | Clone MedThink repo từ GitHub (depth=1) |
| 3 | Hướng dẫn download images (VQA-RAD, SLAKE, PathVQA) |
| 4 | Load JSON data, normalize columns, computed features |
| 5 | Schema inspection + sample records |
| 6 | Dataset size & split distribution |
| 7 | Answer type distribution (Closed/Open) |
| 8 | Top-K answer frequency |
| 9 | Question length distribution |
| 10 | Rationale length (words + sentences) |
| 11 | Question type analysis (WHAT/YES-NO/WHERE…) |
| 12 | **Anatomical region analysis** (R-RAD, small-N warning) |
| 13 | Rationale quality metrics (lexical diversity, medical term coverage) |
| 14 | Rationale word cloud |
| 15 | **Rationale ↔ Answer relationship** |
| 16 | Image statistics (size, mode — nếu images có sẵn) |
| 17 | **Data quality checks** (duplicates, leakage, missing) |
| 18 | **Verifier feasibility pre-analysis** (variance, stratification) |
| 19 | **Threshold strategy decision** (Fixed vs Adaptive analysis) |

**Tiêu chí thành công:**
- Biết chính xác schema của từng dataset (field names, types)
- Hiểu phân phối imbalance → plan stratified sampling cho verifier
- Xác nhận anatomical region info có trong data hay cần join với VQA-RAD metadata
- `eda_outputs/summary_table.csv` sẵn sàng để copy vào thesis

**Thời gian ước tính:** ~1 giờ (CPU, không cần GPU)

---

### Phase 0B: Thiết lập Môi trường & Clone Repo (Notebook 0)

**Mục tiêu:** Chuẩn bị toàn bộ dependencies trên Colab, clone repo MedThink, tải datasets.

#### [NEW] `notebooks/00_setup_environment.ipynb`

```python
# ============================================================
# Cell 1: Mount Drive & Clone MedThink repo
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_ROOT = '/content/drive/MyDrive/MedThink_Verifier'
os.makedirs(PROJECT_ROOT, exist_ok=True)

!git clone https://github.com/Tang-xiaoxiao/Medthink.git /content/Medthink
# Lưu vào Drive để persist qua sessions
!cp -r /content/Medthink {PROJECT_ROOT}/Medthink_repo

# ============================================================
# Cell 2: Cài đặt dependencies
# ============================================================
!pip install -q transformers==4.40.0 datasets accelerate
!pip install -q open_clip_torch  # cho BiomedCLIP
!pip install -q sentence-transformers  # cho semantic similarity
!pip install -q scipy scikit-learn matplotlib seaborn
!pip install -q nltk rouge-score  # cho text metrics

# DETR dependencies (từ MedThink repo)
!pip install -q timm

# ============================================================
# Cell 3: Tải datasets R-RAD, R-SLAKE, R-Path
# ============================================================
# Datasets nằm trong repo Medthink_Dataset/
# Kiểm tra cấu trúc
!ls {PROJECT_ROOT}/Medthink_repo/Medthink_Dataset/

# ============================================================
# Cell 4: Tải BiomedCLIP model (cache vào Drive)
# ============================================================
import open_clip
model_clip, preprocess_clip = open_clip.create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer_clip = open_clip.get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
# Save cache path
print("BiomedCLIP loaded successfully!")

# ============================================================
# Cell 5: Verify GPU
# ============================================================
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Tiêu chí thành công:** Repo cloned, tất cả packages installed, BiomedCLIP loaded, GPU detected.
**Thời gian ước tính:** ~15 phút.

---

### Phase 1: Reproduce MedThink Baseline & Thu thập Predictions (Notebook 1)

**Mục tiêu:** Chạy inference MedThink với chiến lược Reasoning (R→A) trên test set, thu thập tất cả {I, Q, R_pred, A_pred, A_gt}.

#### [NEW] `notebooks/01_reproduce_baseline.ipynb`

```python
# ============================================================
# Cell 1: Load MedThink pretrained model
# ============================================================
# Sử dụng code từ Medthink_Code/ của repo gốc
# Cần adapt paths cho Colab

import sys
sys.path.append(f'{PROJECT_ROOT}/Medthink_repo/Medthink_Code')

# Load config và model theo instructions của repo
# Giả định repo cung cấp pretrained checkpoints hoặc training script
# Nếu không có checkpoint → cần train lại (~8-12 GPU-giờ trên T4)

# ============================================================
# Cell 2: Inference trên 3 datasets với Reasoning strategy
# ============================================================
import json
import pandas as pd

def run_inference_reasoning(model, dataset, dataset_name):
    """
    Chạy inference với Reasoning strategy (R→A).
    Thu thập: image_id, question, R_pred, A_pred, A_gt, R_gt (nếu có)
    """
    results = []
    for sample in dataset:
        image = sample['image']
        question = sample['question']
        answer_gt = sample['answer']
        rationale_gt = sample.get('rationale', '')
        
        # MedThink generates: "rationale: {R} answer: {A}"
        output = model.generate(image, question, strategy='reasoning')
        r_pred, a_pred = parse_reasoning_output(output)
        
        results.append({
            'image_id': sample['image_id'],
            'image_path': sample['image_path'],
            'question': question,
            'answer_gt': answer_gt,
            'answer_pred': a_pred,
            'rationale_gt': rationale_gt,
            'rationale_pred': r_pred,
            'is_correct': (a_pred.strip().lower() == answer_gt.strip().lower())
        })
    
    df = pd.DataFrame(results)
    df.to_csv(f'{PROJECT_ROOT}/predictions_{dataset_name}_reasoning.csv', index=False)
    
    acc = df['is_correct'].mean() * 100
    print(f"{dataset_name} Reasoning Accuracy: {acc:.1f}%")
    return df

# Chạy cho 3 datasets
for ds_name in ['R-RAD', 'R-SLAKE', 'R-Path']:
    df = run_inference_reasoning(model, datasets[ds_name], ds_name)

# ============================================================
# Cell 3: Thống kê baseline
# ============================================================
# Verify: R-RAD Reasoning accuracy ≈ 78.4% (so với 83.5% Explanation)
# → xác nhận gap ~5.1 điểm
for ds_name in ['R-RAD', 'R-SLAKE', 'R-Path']:
    df = pd.read_csv(f'{PROJECT_ROOT}/predictions_{ds_name}_reasoning.csv')
    total = len(df)
    correct = df['is_correct'].sum()
    wrong = total - correct
    print(f"{ds_name}: Total={total}, Correct={correct}, Wrong={wrong}, "
          f"Accuracy={correct/total*100:.1f}%")
```

**Tiêu chí thành công:**
- Accuracy Reasoning strategy gần với paper (cho phép ±1-2% do seed)
- File CSV chứa predictions cho cả 3 datasets
- Cột `is_correct` cho phép chia thành "rationale tốt" vs "rationale xấu"

**Thời gian ước tính:** 2-4 GPU-giờ (nếu cần train lại: +8-12 GPU-giờ)

---

### Phase 2: Xây dựng Bộ Nhãn Yếu (Weak Labels) cho Verifier (Notebook 2)

**Mục tiêu:** Tạo nhãn binary {good_rationale=1, bad_rationale=0} từ ground-truth answer, KHÔNG cần gán nhãn mới.

#### [NEW] `notebooks/02_build_weak_labels.ipynb`

```python
# ============================================================
# Cell 1: Logic gán nhãn yếu
# ============================================================
"""
Chiến lược gán nhãn:
- Nếu A_pred == A_gt → R_pred likely good → label = 1
- Nếu A_pred != A_gt → R_pred likely bad  → label = 0

Giả định chính (cần thảo luận trong paper):
  "Rationale tốt dẫn đến answer đúng, rationale xấu dẫn đến answer sai"
  
Hạn chế đã biết:
  - Right answer, wrong reasoning (lucky guess) → false positive
  - Wrong answer, good reasoning (model limitation) → false negative
  - Đây chính là lý do cần CLIP-based verification bổ sung
"""

import pandas as pd

def create_weak_labels(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # Label 1 = good rationale, 0 = bad rationale
    df['weak_label'] = df['is_correct'].astype(int)
    
    # Statistics
    n_good = df['weak_label'].sum()
    n_bad = len(df) - n_good
    ratio = n_good / n_bad if n_bad > 0 else float('inf')
    
    print(f"Good rationales: {n_good}, Bad rationales: {n_bad}, Ratio: {ratio:.2f}")
    
    # Nếu imbalanced quá → sẽ cần stratified sampling khi train verifier
    df.to_csv(output_path, index=False)
    return df

# ============================================================
# Cell 2: Chia train/val/test cho verifier
# ============================================================
from sklearn.model_selection import train_test_split

def split_verifier_data(df, dataset_name):
    """
    Chia data cho verifier training:
    - 60% train verifier
    - 20% validation (chọn threshold)
    - 20% test (đánh giá final)
    
    Stratified theo weak_label để đảm bảo cân bằng
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.4, stratify=df['weak_label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['weak_label'], random_state=42
    )
    
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = f'{PROJECT_ROOT}/verifier_data/{dataset_name}_{split_name}.csv'
        split_df.to_csv(path, index=False)
        print(f"{dataset_name} {split_name}: {len(split_df)} samples, "
              f"positive rate: {split_df['weak_label'].mean():.2%}")
    
    return train_df, val_df, test_df

# ============================================================
# Cell 3: Phân tích theo vùng giải phẫu (cho R-RAD)
# ============================================================
"""
Bảng 4 trong paper cho thấy lỗi tập trung ở:
- Pelvic Cavity (N=22): accuracy thấp nhất
- Chest (N=5): cỡ mẫu rất nhỏ
→ Cần tag mỗi sample với anatomical region
"""

def tag_anatomical_region(df, metadata_path):
    """
    Sử dụng metadata từ VQA-RAD gốc để map image → anatomical region
    VQA-RAD có annotation: Head, Chest, Abdomen, Pelvic Cavity
    """
    metadata = pd.read_json(metadata_path)
    # Merge anatomical region info
    df = df.merge(metadata[['image_id', 'answer_type', 'question_type', 
                             'image_organ']], on='image_id', how='left')
    
    # Statistics per region
    for region in df['image_organ'].unique():
        region_df = df[df['image_organ'] == region]
        acc = region_df['is_correct'].mean() * 100
        print(f"{region}: N={len(region_df)}, Accuracy={acc:.1f}%")
    
    return df
```

**Tiêu chí thành công:**
- Bộ nhãn yếu cho cả 3 datasets
- Chia train/val/test stratified
- Phân tích per-region cho R-RAD xác nhận pattern từ Bảng 4

**Thời gian ước tính:** ~30 phút (CPU only)

---

### Phase 3: CLIP-based Rationale Verifier (Notebook 3) — Verifier #1

**Mục tiêu:** Đo độ nhất quán ngữ nghĩa giữa rationale R và ảnh I bằng BiomedCLIP.

#### [NEW] `notebooks/03_clip_verifier.ipynb`

```python
# ============================================================
# Cell 1: CLIP Similarity Score
# ============================================================
import torch
import open_clip
from PIL import Image
import numpy as np

class CLIPRationaleVerifier:
    """
    Verifier #1: Dùng BiomedCLIP đo similarity giữa:
    - Ảnh y khoa I
    - Các cụm từ mô tả trong rationale R
    
    Trực giác: Nếu R mô tả "thấy khối u ở phổi" nhưng ảnh là CT bụng
    → CLIP similarity thấp → flag "không chắc chắn"
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model = self.model.to(device).eval()
    
    def compute_similarity(self, image_path, rationale_text):
        """
        Tính cosine similarity giữa image embedding và text embedding.
        """
        # Encode image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode rationale text (truncate nếu quá dài)
        text_tokens = self.tokenizer([rationale_text]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    def compute_chunk_similarities(self, image_path, rationale_text, 
                                     chunk_method='sentences'):
        """
        Chia rationale thành các chunks (câu hoặc cụm từ y khoa),
        tính similarity cho từng chunk, trả về statistics.
        
        Lý do: Một rationale dài có thể chứa 1 câu hallucinated
        giữa nhiều câu đúng → average similarity vẫn cao.
        Cần xem min similarity.
        """
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        
        if chunk_method == 'sentences':
            chunks = sent_tokenize(rationale_text)
        else:
            # Chia theo cụm từ ngắn (mỗi ~10 từ)
            words = rationale_text.split()
            chunks = [' '.join(words[i:i+10]) for i in range(0, len(words), 10)]
        
        if not chunks:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        sims = []
        for chunk in chunks:
            if len(chunk.strip()) > 5:  # skip chunks quá ngắn
                sim = self.compute_similarity(image_path, chunk)
                sims.append(sim)
        
        if not sims:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        return {
            'mean': np.mean(sims),
            'min': np.min(sims),
            'max': np.max(sims),
            'std': np.std(sims),
            'n_chunks': len(sims),
            'chunk_sims': sims
        }

# ============================================================
# Cell 2: Chạy CLIP Verifier trên toàn bộ predictions
# ============================================================
verifier = CLIPRationaleVerifier(device='cuda')

def score_all_predictions(csv_path, output_path, image_dir):
    df = pd.read_csv(csv_path)
    
    all_scores = []
    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, row['image_path'])
        rationale = row['rationale_pred']
        
        # Full rationale similarity
        full_sim = verifier.compute_similarity(image_path, rationale)
        
        # Chunk-level similarity
        chunk_stats = verifier.compute_chunk_similarities(image_path, rationale)
        
        scores = {
            'clip_full_sim': full_sim,
            'clip_chunk_mean': chunk_stats['mean'],
            'clip_chunk_min': chunk_stats['min'],
            'clip_chunk_std': chunk_stats['std'],
        }
        all_scores.append(scores)
        
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(df)}")
    
    scores_df = pd.DataFrame(all_scores)
    result_df = pd.concat([df, scores_df], axis=1)
    result_df.to_csv(output_path, index=False)
    return result_df

# ============================================================
# Cell 3: Phân tích phân biệt (Discrimination Analysis)
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

def analyze_clip_discrimination(df, score_col='clip_full_sim'):
    """
    Câu hỏi chính: CLIP similarity có phân biệt được 
    rationale đúng vs sai không?
    """
    good = df[df['weak_label'] == 1][score_col]
    bad = df[df['weak_label'] == 0][score_col]
    
    print(f"Good rationales - mean: {good.mean():.4f}, std: {good.std():.4f}")
    print(f"Bad rationales  - mean: {bad.mean():.4f}, std: {bad.std():.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((good.std()**2 + bad.std()**2) / 2)
    cohens_d = (good.mean() - bad.mean()) / pooled_std
    print(f"Cohen's d: {cohens_d:.3f}")
    
    # AUC-ROC
    auc = roc_auc_score(df['weak_label'], df[score_col])
    print(f"AUC-ROC: {auc:.3f}")
    
    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(good, bins=30, alpha=0.6, label='Good R', color='green')
    axes[0].hist(bad, bins=30, alpha=0.6, label='Bad R', color='red')
    axes[0].set_xlabel(score_col)
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].set_title(f'Distribution of {score_col}')
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(df['weak_label'], df[score_col])
    axes[1].plot(fpr, tpr, label=f'AUC={auc:.3f}')
    axes[1].plot([0,1], [0,1], 'k--')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{PROJECT_ROOT}/clip_discrimination.png', dpi=150)
    plt.show()
    
    return auc, cohens_d
```

**Tiêu chí thành công:**
- AUC-ROC > 0.55 (tốt hơn random) → CLIP có tín hiệu phân biệt
- AUC-ROC > 0.65 → kết quả tốt, đủ cho publication
- Cohen's d > 0.2 (small effect) → có sự khác biệt có ý nghĩa

**Thời gian ước tính:** 3-5 GPU-giờ cho 3 datasets

---

### Phase 4: Self-Consistency Verifier bằng Semantic Entropy (Notebook 4) — Verifier #2

**Mục tiêu:** Sinh rationale nhiều lần với sampling, đo độ phân tán ngữ nghĩa. Rationale hallucinated → không ổn định qua các lần sampling.

#### [NEW] `notebooks/04_self_consistency_verifier.ipynb`

```python
# ============================================================
# Cell 1: Multi-Sample Generation
# ============================================================
import torch
from sentence_transformers import SentenceTransformer

class SelfConsistencyVerifier:
    """
    Verifier #2: Semantic Entropy / Self-Consistency
    
    Nguyên lý:
    - Sinh rationale K lần (K=5-10) với temperature sampling
    - Nếu model "hiểu" ảnh → các rationale sẽ nhất quán về ngữ nghĩa
    - Nếu model "hallucinate" → các rationale sẽ phân tán (semantic entropy cao)
    
    Lấy cảm hứng từ:
    - VASE (Vision-Amplified Semantic Entropy, MICCAI 2025)
    - Semantic Entropy Probes (ICML 2024)
    """
    
    def __init__(self, medthink_model, n_samples=5, temperature=0.7,
                 sim_model_name='all-MiniLM-L6-v2'):
        self.medthink = medthink_model
        self.n_samples = n_samples
        self.temperature = temperature
        
        # Sentence-BERT for semantic similarity between rationales
        # Nhẹ (~22M params) và hiệu quả
        self.sent_model = SentenceTransformer(sim_model_name)
    
    def generate_multiple_rationales(self, image, question):
        """
        Sinh K rationales với temperature sampling.
        """
        rationales = []
        answers = []
        
        for _ in range(self.n_samples):
            output = self.medthink.generate(
                image, question,
                strategy='reasoning',
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9
            )
            r, a = parse_reasoning_output(output)
            rationales.append(r)
            answers.append(a)
        
        return rationales, answers
    
    def compute_semantic_entropy(self, rationales):
        """
        Tính semantic entropy:
        1. Encode tất cả rationales thành embeddings
        2. Tính pairwise cosine similarity
        3. Semantic entropy ≈ 1 - mean(pairwise_similarity)
        
        Cao → không nhất quán → likely hallucination
        Thấp → nhất quán → likely reliable
        """
        if len(rationales) < 2:
            return {'entropy': 0.0, 'mean_sim': 1.0}
        
        embeddings = self.sent_model.encode(rationales)
        
        # Pairwise cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        
        # Lấy upper triangle (excluding diagonal)
        n = len(rationales)
        pairwise_sims = []
        for i in range(n):
            for j in range(i+1, n):
                pairwise_sims.append(sim_matrix[i][j])
        
        mean_sim = np.mean(pairwise_sims)
        std_sim = np.std(pairwise_sims)
        min_sim = np.min(pairwise_sims)
        
        # Semantic entropy (higher = more uncertain)
        semantic_entropy = 1.0 - mean_sim
        
        return {
            'semantic_entropy': semantic_entropy,
            'mean_pairwise_sim': mean_sim,
            'std_pairwise_sim': std_sim,
            'min_pairwise_sim': min_sim,
        }
    
    def compute_answer_consistency(self, answers):
        """
        Answer consistency: tỷ lệ answer phổ biến nhất.
        
        Ví dụ: 5 lần sampling, 4 lần answer "yes", 1 lần "no"
        → consistency = 4/5 = 0.8
        """
        from collections import Counter
        counter = Counter([a.strip().lower() for a in answers])
        most_common_count = counter.most_common(1)[0][1]
        consistency = most_common_count / len(answers)
        
        return {
            'answer_consistency': consistency,
            'n_unique_answers': len(counter),
            'majority_answer': counter.most_common(1)[0][0],
        }
    
    def score_sample(self, image, question):
        """
        Tổng hợp tất cả metrics cho 1 sample.
        """
        rationales, answers = self.generate_multiple_rationales(image, question)
        
        entropy_scores = self.compute_semantic_entropy(rationales)
        consistency_scores = self.compute_answer_consistency(answers)
        
        return {**entropy_scores, **consistency_scores}

# ============================================================
# Cell 2: Chạy Self-Consistency trên toàn bộ test set
# ============================================================
sc_verifier = SelfConsistencyVerifier(
    medthink_model=model, 
    n_samples=5,       # 5 lần sampling — balance giữa quality và compute
    temperature=0.7    # Moderate diversity
)

def score_all_self_consistency(csv_path, output_path, image_dir):
    df = pd.read_csv(csv_path)
    
    all_scores = []
    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, row['image_path'])
        question = row['question']
        image = load_image(image_path)  # Load theo format MedThink
        
        scores = sc_verifier.score_sample(image, question)
        all_scores.append(scores)
        
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(df)} - "
                  f"entropy: {scores['semantic_entropy']:.3f}, "
                  f"consistency: {scores['answer_consistency']:.2f}")
    
    scores_df = pd.DataFrame(all_scores)
    result_df = pd.concat([df, scores_df], axis=1)
    result_df.to_csv(output_path, index=False)
    return result_df

# ============================================================
# Cell 3: Phân tích (tương tự Cell 3 Phase 3)
# ============================================================
# AUC-ROC cho semantic_entropy vs weak_label
# Lưu ý: entropy CAO → likely BAD, nên invert khi tính AUC
# auc = roc_auc_score(df['weak_label'], -df['semantic_entropy'])
```

**Tiêu chí thành công:**
- Semantic entropy có phân phối khác biệt giữa good/bad rationales
- AUC-ROC > 0.55 (sử dụng negative entropy)
- Answer consistency mạnh hơn → xác nhận self-consistency hoạt động

**Thời gian ước tính:** 10-15 GPU-giờ (mỗi sample cần K=5 forward passes)

> [!WARNING]
> Phase 4 là **bottleneck compute lớn nhất**. Nếu ngân sách hạn chế:
> - Giảm K từ 5 xuống 3
> - Chỉ chạy trên R-RAD trước (nhỏ nhất, ~451 samples), nếu có tín hiệu tốt mới mở rộng
> - Hoặc chỉ chạy trên test split của verifier data (~20% tổng)

---

### Phase 5: Ensemble Verifier & Confidence Scoring (Notebook 5)

**Mục tiêu:** Kết hợp CLIP verifier + Self-Consistency verifier thành ensemble, tạo confidence score và flag.

#### [NEW] `notebooks/05_ensemble_verifier.ipynb`

```python
# ============================================================
# Cell 1: Feature Engineering & Ensemble
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score, 
                              precision_recall_curve, f1_score)

class EnsembleVerifier:
    """
    Kết hợp features từ cả 2 verifiers:
    
    CLIP features:
    - clip_full_sim: similarity toàn bộ rationale
    - clip_chunk_min: min chunk similarity (detect hallucinated sentences)
    - clip_chunk_std: variance giữa các chunks
    
    Self-Consistency features:
    - semantic_entropy: phân tán ngữ nghĩa giữa K samples
    - answer_consistency: tỷ lệ answer thống nhất
    - min_pairwise_sim: worst-case pairwise similarity
    
    Classifier: Logistic Regression (interpretable, ít overfitting với N nhỏ)
    """
    
    FEATURE_COLS = [
        'clip_full_sim', 'clip_chunk_min', 'clip_chunk_std',
        'semantic_entropy', 'answer_consistency', 'min_pairwise_sim'
    ]
    
    def __init__(self, classifier_type='logistic'):
        self.scaler = StandardScaler()
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                C=1.0, class_weight='balanced', random_state=42
            )
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=100, class_weight='balanced', 
                max_depth=3, random_state=42  # shallow to avoid overfit
            )
    
    def prepare_features(self, df):
        X = df[self.FEATURE_COLS].fillna(0).values
        y = df['weak_label'].values
        return X, y
    
    def train(self, train_df, val_df):
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train
        self.classifier.fit(X_train_scaled, y_train)
        
        # Validate
        val_probs = self.classifier.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        
        # Find optimal threshold on validation set
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1s)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        print(f"Validation AUC: {val_auc:.3f}")
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")
        print(f"At threshold - Precision: {precisions[optimal_idx]:.3f}, "
              f"Recall: {recalls[optimal_idx]:.3f}, F1: {f1s[optimal_idx]:.3f}")
        
        # Feature importance (cho Logistic Regression)
        if hasattr(self.classifier, 'coef_'):
            importances = pd.DataFrame({
                'feature': self.FEATURE_COLS,
                'coefficient': self.classifier.coef_[0],
                'abs_coefficient': np.abs(self.classifier.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            print("\nFeature Importance:")
            print(importances.to_string(index=False))
        
        return val_auc
    
    def predict(self, df):
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        probs = self.classifier.predict_proba(X_scaled)[:, 1]
        flags = (probs < self.optimal_threshold).astype(int)  # 1 = uncertain
        
        df = df.copy()
        df['confidence_score'] = probs
        df['uncertainty_flag'] = flags
        
        return df

# ============================================================
# Cell 2: Train & Evaluate Ensemble
# ============================================================
def evaluate_ensemble(dataset_name):
    # Load merged data (CLIP + Self-Consistency scores)
    train_df = pd.read_csv(f'{PROJECT_ROOT}/merged_{dataset_name}_train.csv')
    val_df = pd.read_csv(f'{PROJECT_ROOT}/merged_{dataset_name}_val.csv')
    test_df = pd.read_csv(f'{PROJECT_ROOT}/merged_{dataset_name}_test.csv')
    
    ensemble = EnsembleVerifier(classifier_type='logistic')
    val_auc = ensemble.train(train_df, val_df)
    
    # Evaluate on test set
    test_result = ensemble.predict(test_df)
    
    # Metrics
    test_auc = roc_auc_score(
        test_result['weak_label'], 
        test_result['confidence_score']
    )
    
    # Flagged samples analysis
    flagged = test_result[test_result['uncertainty_flag'] == 1]
    unflagged = test_result[test_result['uncertainty_flag'] == 0]
    
    flagged_error_rate = 1 - flagged['is_correct'].mean() if len(flagged) > 0 else 0
    unflagged_error_rate = 1 - unflagged['is_correct'].mean() if len(unflagged) > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"ENSEMBLE RESULTS - {dataset_name}")
    print(f"{'='*50}")
    print(f"Test AUC-ROC: {test_auc:.3f}")
    print(f"Flagged: {len(flagged)} samples ({len(flagged)/len(test_result)*100:.1f}%)")
    print(f"  → Error rate in flagged: {flagged_error_rate:.1%}")
    print(f"Unflagged: {len(unflagged)} samples ({len(unflagged)/len(test_result)*100:.1f}%)")
    print(f"  → Error rate in unflagged: {unflagged_error_rate:.1%}")
    print(f"Error rate reduction: {flagged_error_rate - unflagged_error_rate:.1%}")
    
    return test_result, ensemble

# ============================================================
# Cell 3: Ablation Study - từng verifier riêng lẻ vs ensemble
# ============================================================
def ablation_study(dataset_name):
    """
    So sánh:
    1. CLIP-only verifier
    2. Self-Consistency-only verifier  
    3. Ensemble (cả 2)
    4. Random baseline
    """
    results = {}
    
    for config_name, features in [
        ('CLIP-only', ['clip_full_sim', 'clip_chunk_min', 'clip_chunk_std']),
        ('SC-only', ['semantic_entropy', 'answer_consistency', 'min_pairwise_sim']),
        ('Ensemble', EnsembleVerifier.FEATURE_COLS),
    ]:
        # Train with subset of features
        ensemble = EnsembleVerifier()
        ensemble.FEATURE_COLS = features
        # ... train and evaluate ...
        results[config_name] = test_auc
    
    # Random baseline
    results['Random'] = 0.5
    
    # Bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['#2196F3', '#FF9800', '#4CAF50', '#9E9E9E'])
    plt.ylabel('AUC-ROC')
    plt.title(f'Ablation Study - {dataset_name}')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random')
    plt.ylim(0.4, 1.0)
    plt.savefig(f'{PROJECT_ROOT}/ablation_{dataset_name}.png', dpi=150)
    plt.show()
```

**Tiêu chí thành công:**
- Ensemble AUC > max(CLIP-only, SC-only) → kết hợp có ích
- Error rate trong flagged samples >> error rate trong unflagged samples
- Feature importance cho thấy cả 2 loại features đều đóng góp

**Thời gian ước tính:** 1-2 giờ (chủ yếu CPU)

---

### Phase 6: Phân tích Theo Vùng Giải phẫu & Case Study (Notebook 6)

**Mục tiêu:** Xác nhận verifier hoạt động đặc biệt tốt ở các vùng mà MedThink hay sai (Bảng 4), tạo qualitative examples.

#### [NEW] `notebooks/06_anatomical_analysis.ipynb`

```python
# ============================================================
# Cell 1: Per-Region Analysis
# ============================================================
def per_region_analysis(result_df, dataset_name='R-RAD'):
    """
    Phân tích verifier performance theo vùng giải phẫu.
    Trọng tâm: Pelvic Cavity, Abdomen (nơi MedThink hay sai nhất)
    """
    regions = result_df['image_organ'].unique()
    
    region_results = []
    for region in regions:
        region_df = result_df[result_df['image_organ'] == region]
        n = len(region_df)
        
        if n < 5:  # Quá ít để phân tích
            print(f"⚠️ {region}: N={n} — too few for reliable analysis")
            continue
        
        # MedThink baseline accuracy
        baseline_acc = region_df['is_correct'].mean()
        
        # Verifier AUC (nếu đủ cả 2 class)
        if region_df['weak_label'].nunique() > 1:
            auc = roc_auc_score(region_df['weak_label'], 
                               region_df['confidence_score'])
        else:
            auc = float('nan')
        
        # Flagging rate
        flag_rate = region_df['uncertainty_flag'].mean()
        
        # "Selective accuracy" = accuracy nếu chỉ trả lời unflagged samples
        unflagged = region_df[region_df['uncertainty_flag'] == 0]
        selective_acc = unflagged['is_correct'].mean() if len(unflagged) > 0 else 0
        coverage = len(unflagged) / n
        
        region_results.append({
            'Region': region,
            'N': n,
            'Baseline Acc': f'{baseline_acc:.1%}',
            'Verifier AUC': f'{auc:.3f}' if not np.isnan(auc) else 'N/A',
            'Flag Rate': f'{flag_rate:.1%}',
            'Selective Acc': f'{selective_acc:.1%}',
            'Coverage': f'{coverage:.1%}',
        })
    
    results_table = pd.DataFrame(region_results)
    print(results_table.to_string(index=False))
    results_table.to_csv(f'{PROJECT_ROOT}/per_region_{dataset_name}.csv', index=False)
    return results_table

# ============================================================
# Cell 2: Bootstrap Confidence Intervals
# ============================================================
def bootstrap_ci(data, metric_fn, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap CI cho small N (đặc biệt Pelvic Cavity N=22, Chest N=5)
    """
    stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        stats.append(metric_fn(sample))
    
    lower = np.percentile(stats, (1-ci)/2 * 100)
    upper = np.percentile(stats, (1+ci)/2 * 100)
    mean = np.mean(stats)
    
    return mean, lower, upper

# ============================================================
# Cell 3: Qualitative Case Studies
# ============================================================
def generate_case_studies(result_df, n_cases=5):
    """
    Chọn các ca minh họa:
    1. True Positive: Verifier đúng flag rationale sai
    2. True Negative: Verifier đúng không flag rationale tốt
    3. False Positive: Verifier sai flag rationale tốt (conservative)
    4. False Negative: Verifier bỏ sót rationale sai (dangerous)
    
    Cho mỗi ca: hiển thị ảnh, câu hỏi, rationale, answer, scores
    """
    cases = {
        'True Positive (Correct Flag)': 
            result_df[(result_df['uncertainty_flag'] == 1) & 
                      (result_df['is_correct'] == False)],
        'True Negative (Correct Trust)': 
            result_df[(result_df['uncertainty_flag'] == 0) & 
                      (result_df['is_correct'] == True)],
        'False Positive (Over-cautious)': 
            result_df[(result_df['uncertainty_flag'] == 1) & 
                      (result_df['is_correct'] == True)],
        'False Negative (Missed Error)': 
            result_df[(result_df['uncertainty_flag'] == 0) & 
                      (result_df['is_correct'] == False)],
    }
    
    for case_type, case_df in cases.items():
        print(f"\n{'='*60}")
        print(f"📋 {case_type}: {len(case_df)} samples")
        print(f"{'='*60}")
        
        for _, row in case_df.head(n_cases).iterrows():
            print(f"\n  Image: {row['image_path']}")
            print(f"  Question: {row['question']}")
            print(f"  Rationale: {row['rationale_pred'][:200]}...")
            print(f"  Answer GT: {row['answer_gt']}, Pred: {row['answer_pred']}")
            print(f"  CLIP sim: {row['clip_full_sim']:.3f}, "
                  f"Entropy: {row['semantic_entropy']:.3f}, "
                  f"Confidence: {row['confidence_score']:.3f}")
```

**Tiêu chí thành công:**
- Selective accuracy > baseline accuracy (bác sĩ tin hơn khi verifier không flag)
- Case studies cho thấy verifier catch được ≥1 lỗi rõ ràng ở Pelvic/Abdomen
- Bootstrap CIs không chứa 0 (cho AUC difference)

**Thời gian ước tính:** 1-2 giờ (CPU)

---

### Phase 7: Tổng hợp & Báo cáo (Notebook 7)

#### [NEW] `notebooks/07_final_report.ipynb`

```python
# ============================================================
# Cell 1: Summary Table (Table cho khóa luận)
# ============================================================
"""
Bảng kết quả chính:

| Dataset | MedThink Acc | Verifier AUC | Flag Rate | Selective Acc | Coverage |
|---------|-------------|-------------|-----------|---------------|----------|
| R-RAD   | ~78.4%      | ???         | ???       | ???           | ???      |
| R-SLAKE | ~??%        | ???         | ???       | ???           | ???      |
| R-Path  | ~??%        | ???         | ???       | ???           | ???      |

Bảng ablation:
| Method          | R-RAD AUC | R-SLAKE AUC | R-Path AUC |
|-----------------|-----------|-------------|------------|
| Random          | 0.500     | 0.500       | 0.500      |
| CLIP-only       | ???       | ???         | ???        |
| SC-only         | ???       | ???         | ???        |
| Ensemble        | ???       | ???         | ???        |
"""

# ============================================================
# Cell 2: Tính "Trust Improvement"
# ============================================================
def compute_trust_metrics(result_df):
    """
    Metrics chính cho phần "Trust and Reliability":
    
    1. Error Detection Rate: % lỗi bị catch bởi flag
    2. False Alarm Rate: % sample đúng bị flag nhầm
    3. Risk Reduction: error rate(unflagged) - error rate(all)
    4. ECE (Expected Calibration Error): confidence score calibrated?
    """
    total_errors = len(result_df[result_df['is_correct'] == False])
    flagged_errors = len(result_df[(result_df['uncertainty_flag'] == 1) & 
                                   (result_df['is_correct'] == False)])
    
    error_detection_rate = flagged_errors / total_errors if total_errors > 0 else 0
    
    total_correct = len(result_df[result_df['is_correct'] == True])
    flagged_correct = len(result_df[(result_df['uncertainty_flag'] == 1) & 
                                    (result_df['is_correct'] == True)])
    false_alarm_rate = flagged_correct / total_correct if total_correct > 0 else 0
    
    all_error_rate = 1 - result_df['is_correct'].mean()
    unflagged = result_df[result_df['uncertainty_flag'] == 0]
    unflagged_error_rate = 1 - unflagged['is_correct'].mean() if len(unflagged) > 0 else 0
    risk_reduction = all_error_rate - unflagged_error_rate
    
    print(f"Error Detection Rate: {error_detection_rate:.1%}")
    print(f"False Alarm Rate: {false_alarm_rate:.1%}")
    print(f"Risk Reduction: {risk_reduction:.1%} absolute")
    
    return {
        'error_detection_rate': error_detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'risk_reduction': risk_reduction
    }
```

---

## Tóm tắt Notebooks & Dependencies

| Phase | Notebook | Mục tiêu | GPU? | Thời gian |
|-------|----------|----------|------|-----------|
| 0A | `00_download_and_eda` ⭐ | Download + EDA đầy đủ | ❌ | 1h |
| 0B | `00b_setup_environment` | Setup, clone, install | ❌ | 15 phút |
| 1 | `01_reproduce_baseline` | Reproduce **cả 2 chiến lược** | ✅ | 3-6h |
| 2 | `02_build_weak_labels` | Tạo labels cho verifier | ❌ | 30 phút |
| 3 | `03_clip_verifier` | BiomedCLIP verification | ✅ | 3-5h |
| 4 | `04_self_consistency_verifier` | Semantic entropy | ✅ | 10-15h |
| 5 | `05_ensemble_verifier` | Ensemble + threshold analysis | ❌ | 2-3h |
| 6 | `06_anatomical_analysis` | Per-region + case study | ❌ | 1-2h |
| 7 | `07_final_report` | Tổng hợp metrics | ❌ | 1h |

**Tổng GPU-giờ ước tính:** 15-24h (không tính train lại MedThink)

---

## Verification Plan

### Automated Tests
```bash
# Sanity checks trong mỗi notebook:
# 1. MedThink baseline accuracy ≈ paper ± 2%
# 2. Weak label distribution matches expected error rate
# 3. CLIP similarity scores trong range hợp lý [0, 1]
# 4. Semantic entropy ≥ 0
# 5. AUC-ROC > 0.5 (tốt hơn random — test thống kê)
# 6. Ensemble AUC ≥ max(individual AUCs)
```

### Manual Verification
- So sánh accuracy baseline với Table 3 trong paper
- Kiểm tra case studies: rationale flagged có thực sự sai về mặt y khoa không?
- Verify per-region results: Pelvic Cavity có flag rate cao hơn Head không?

---

## Chiến lược Giảm thiểu Rủi ro

| Rủi ro | Xác suất | Giải pháp |
|--------|----------|-----------|
| CLIP không phân biệt R tốt/xấu | Trung bình | → Chuyển sang dùng chunk-level min similarity thay vì full-text |
| Self-consistency quá tốn compute | Cao | → Giảm K=3, chỉ chạy trên R-RAD trước |
| N quá nhỏ cho per-region analysis | Cao | → Gộp region thành Thoracic vs Abdominal vs Other |
| Negative result (verifier ~ random) | Thấp-TB | → Phân tích WHY (rationale format quá uniform → entropy thấp cả đúng/sai) → vẫn là contribution |
| Không reproduce được MedThink accuracy | Thấp | → Liên hệ authors qua GitHub Issues, dùng checkpoint nếu có |

---

## Timeline đề xuất (12 tuần)

| Tuần | Phases | Milestone |
|------|--------|-----------|
| 1-2 | 0, 1 | Reproduce baseline, confirm accuracy gap |
| 3 | 2 | Weak labels ready, data splits done |
| 4-5 | 3 | CLIP verifier complete, initial AUC results |
| 6-8 | 4 | Self-consistency verifier complete |
| 9 | 5 | Ensemble trained, ablation done |
| 10 | 6 | Per-region analysis, case studies |
| 11-12 | 7 | Final report, thesis writing |
