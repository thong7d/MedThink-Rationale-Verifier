import os
import sys

# Force utf-8 encoding for stdout/stderr
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("Downloading kernel output...")
api.kernels_output("phamtohng/medthink-phase-1-clean-baseline", "kaggle_outputs_v17", force=True, quiet=True)
print("Done!")
