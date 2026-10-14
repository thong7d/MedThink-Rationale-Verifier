# =============================================================================
# MedThink Dataset — Master Runner & Guidance
# Script: 00_download_and_eda.py
# =============================================================================
# 1. 00a_download_and_eda.py   -> Chạy trên LOCAL (Python script thuần)
# 2. 00b_threshold_strategy.py -> Chạy trên GOOGLE COLAB (Copy cells vào Colab)
# =============================================================================

import sys
import subprocess
from pathlib import Path

print("""
=============================================================================
📌 CẤU TRÚC SCRIPTS VÀ PHÂN CHIA MÔI TRƯỜNG:

  1. 00a_download_and_eda.py  [MÔI TRƯỜNG LOCAL]
     - Chạy thuần Python trên máy tính local.
     - Tải repository GitHub MedThink, thực hiện toàn bộ EDA 13 bước, xuất biểu đồ & CSV.
     - Lệnh chạy local: python 00a_download_and_eda.py

  2. 00b_threshold_strategy.py [MÔI TRƯỜNG GOOGLE COLAB]
     - Dùng trên Google Colab GPU/CPU.
     - Phân tích chiến lược Fixed vs Adaptive Threshold & học ngưỡng tối ưu F1.
     - Cách dùng: Mở trên Google Colab và copy từng CELL vào Notebook.
=============================================================================
""")

if __name__ == '__main__':
    root = Path(__file__).parent
    script_a = root / "00a_download_and_eda.py"
    if script_a.exists():
        print("[RUNNING LOCAL] Tự động chạy 00a_download_and_eda.py trên máy local...\n")
        subprocess.run([sys.executable, str(script_a)])
