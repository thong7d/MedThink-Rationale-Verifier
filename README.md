# 🩺 MedThink Replication & Rationale Verifier (Graduation Thesis)

> **Đề tài Khóa luận Tốt nghiệp:** Chẩn đoán & Khắc phục Sự Bất ổn của Chiến lược "Reasoning" bằng Cơ chế Tự-Xác Thực Rationale cho MedThink trong Medical Visual Question Answering (Med-VQA).  
> **Reference Paper:** *MedThink: A Rationale-Guided Framework for Explaining Medical Visual Question Answering* (Gai et al., NAACL Findings 2025).

---

## 📌 1. Bối cảnh & Mục tiêu Nghiên cứu

### 1.1. Vấn đề cốt lõi của MedThink (Paper gốc)
- MedThink là mô hình sinh đa phương thức kết hợp **UnifiedQA (T5-base)** và **DETR features** để giải bài toán Medical VQA kèm lời giải thích (Rationale):
  - **Chiến lược Explanation ($A \rightarrow R$):** Sinh câu trả lời trước, sau đó sinh giải thích. Đạt độ chính xác cao: ~83.5% (R-RAD), ~86.3% (R-SLAKE).
  - **Chiến lược Reasoning ($R \rightarrow A$):** Sinh giải thích trước để dẫn dắt câu trả lời. Độ chính xác bị **suy giảm nghiêm trọng** (~5.1 điểm % trên R-RAD) do hiện tượng **ảo giác (hallucination)** và lỗi lập luận lan truyền sang câu trả lời.
- Paper gốc thừa nhận hạn chế về tính đáng tin cậy (*Trust & Reliability*), nhưng **chưa có cơ chế phát hiện khi nào Rationale bị sai**.

### 1.2. Đóng góp của đề tài
1. **Faithful Reproduction:** Tái hiện trung thực pipeline huấn luyện và đánh giá của tác giả MedThink (150 epochs, Multimodal Seq2SeqTrainer, DETR features).
2. **Rationale Verifier (Phase 2):** Xây dựng module tự-kiểm định nhẹ (*post-hoc verifier*) kết hợp **BiomedCLIP** để gán điểm tin cậy (*Confidence Score* / *Uncertainty Flag*) cho từng cặp $\{R, A\}$, giúp người dùng/bác sĩ biết: *"Khi nào KHÔNG NÊN tin vào lời giải thích của mô hình?"*.

---

## 🚀 2. Những công việc ĐÃ HOÀN THÀNH (What Has Been Done)

### ✅ Phase 0: Phân tích Dữ liệu (EDA) & Thiết kế Kế hoạch
- Khảo sát và tiền xử lý 3 bộ dữ liệu chuẩn: **R-RAD**, **R-SLAKE**, và **R-PathVQA** (cả Closed-end và Open-end).
- Lập tài liệu kế hoạch thực nghiệm 8 giai đoạn chi tiết: [`implementation_plan.md`](./implementation_plan.md).

### ✅ Phase 1: Tái hiện Mô hình Gốc & Giải quyết các Lỗi Kỹ thuật Nền tảng
Quá trình replicate phát hiện và đã sửa đổi triệt để các vấn đề sau:
1. **Lỗi Tràn/Lệch DynamicCache (`Transformers >= 4.40`):**
   - *Triệu chứng:* Khi inference sinh text, T5 ném lỗi kích thước Cache mismatch (1024 vs 512).
   - *Khắc phục:* Bổ sung cấu hình `decoder_config.is_encoder_decoder = True` đồng bộ trong `T5ForMultimodalGeneration`.
2. **Lỗi Ảnh Git LFS trên Môi trường Cloud (Kaggle/Colab):**
   - *Triệu chứng:* `git clone --depth 1` từ repo gốc chỉ tải về pointer file text (~130 bytes) thay vì ảnh nhị phân thật, dẫn đến `PIL.UnidentifiedImageError`.
   - *Khắc phục:* Tự động hóa khâu pull Git LFS hoặc load trực tiếp ảnh chuẩn trước khi trích xuất DETR features.
3. **Hiện tượng Lặp từ vô tận & Accuracy thấp (Repetition Loop):**
   - *Triệu chứng:* Ban đầu model sinh chuỗi `((((` hoặc `aaaa...` làm Accuracy rớt xuống 0% - 7%.
   - *Khắc phục:* Đồng bộ đúng `ClosedMedVQADataset` theo tác giả, điều chỉnh `max_target_len = 256`, thiết lập tham số sinh `repetition_penalty` và huấn luyện đủ **150 Epochs**.
4. **Tùy biến `MultimodalSeq2SeqTrainer`:**
   - HuggingFace `Seq2SeqTrainer` mặc định không hỗ trợ truyền `image_ids` vào visual encoder. Đã override hàm `compute_loss` và `prediction_step` để xử lý mượt mà feature DETR.

### 🔄 Phase 2: Rationale Verifier (Đang triển khai)
- Xây dựng pipeline tạo nhãn yếu (*Weak Label Generation*) dựa trên chất lượng rationale: [`phase2/build_weak_labels.py`](./phase2/build_weak_labels.py).
- Tích hợp **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) để đo độ tương đồng ngữ nghĩa giữa ảnh X-quang/CT và rationale sinh ra.

---

## 📂 3. Cấu trúc Repository

```text
├── README.md                     # Tài liệu tổng quan (file này)
├── implementation_plan.md        # Kế hoạch chi tiết 8 Phase của khóa luận
├── 00_download_and_eda.py        # Script tải và phân tích dữ liệu EDA
├── phase1/                       # Mã nguồn Phase 1 (Replication & Baseline)
│   ├── model.py                  # Kiến trúc MedThink T5 đa phương thức
│   ├── dataset.py                # DataLoader Closed & Open MedVQA
│   ├── extract_detr_features.py  # Trích xuất visual features bằng DETR
│   ├── train.py                  # Script huấn luyện Seq2SeqTrainer
│   ├── evaluate.py               # Script đánh giá Accuracy và Rationale
│   ├── run_phase1_pipeline.py    # Pipeline chạy tự động toàn bộ
│   └── utils.py                  # Tiện ích bổ trợ
├── phase2/                       # Mã nguồn Phase 2 (Rationale Verifier)
│   ├── README.md                 # Tài liệu thiết kế module Verifier
│   └── build_weak_labels.py      # Sinh nhãn phân loại độ tin cậy của rationale
├── kaggle_push/                  # Scripts đóng gói chạy tự động trên Kaggle GPU
│   └── medthink_phase1_kaggle.py # Script độc lập chạy 150 epochs trên Kaggle
└── notebooks/                    # Jupyter Notebooks phục vụ EDA và thử nghiệm
```

---

## 🛠️ 4. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### 4.1. Thiết lập Môi trường Local

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Cài đặt các thư viện phụ thuộc chính
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.0 datasets accelerate
pip install open_clip_torch pandas scikit-learn pillow tqdm
```

### 4.2. Chạy Huấn luyện Phase 1 (150 Epochs)

```bash
# Trích xuất DETR feature từ ảnh
python phase1/extract_detr_features.py --dataset rad --image_dir ./data/R-RAD/images

# Chạy huấn luyện chiến lược Reasoning (R -> A)
python phase1/train.py \
    --dataset rad \
    --strategy reasoning \
    --epochs 150 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 5e-4
```

### 4.3. Chạy trên Kaggle / Google Colab
- Sử dụng script trọn gói tại [`kaggle_push/medthink_phase1_kaggle.py`](./kaggle_push/medthink_phase1_kaggle.py).
- Script đã được tích hợp sẵn:
  - Tự clone dependencies và DETR weights.
  - Tự động download và xử lý ảnh R-RAD / R-SLAKE.
  - Tự động lưu checkpoint và ghi nhận log chi tiết.

---

## 🎯 5. Kế hoạch Tiếp theo (Roadmap cho Nhóm)

1. **Hoàn thiện Phase 1 Runs:** Thu thập kết quả chuẩn 150 epochs cho cả 2 chiến lược `Reasoning` và `Explanation` trên R-RAD & R-SLAKE.
2. **Huấn luyện Verifier (Phase 2):** Sử dụng đặc trưng BiomedCLIP để phân loại Rationale đáng tin vs Rationale ảo giác.
3. **Đánh giá Lâm sàng (Evaluation):** Tính AUROC phát hiện lỗi, đo lường sự cải thiện Accuracy khi loại bỏ các câu trả lời có rationale bị gán cờ rủi ro (*uncertainty flag*).
4. **Viết Báo cáo & Khóa luận:** Hoàn thiện bản thảo khóa luận với bảng số liệu so sánh chi tiết.

---

*Mọi thắc mắc và thảo luận kỹ thuật, vui lòng tạo Issue trên GitHub hoặc trao đổi trực tiếp trong nhóm.*
