import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (ACTIVITY_BILSTM_PATH, ACTIVITY_CNN_LSTM_PATH, BERT_MODEL_DIR,
                        MASK_CNN_PATH, MASK_MOBILENET_PATH, SAFETY_ANN_PATH, SAFETY_SCALER_PATH)

CHECKS = [
    ("02", MASK_CNN_PATH),
    ("03", MASK_MOBILENET_PATH),
    ("04", ACTIVITY_CNN_LSTM_PATH),
    ("05", ACTIVITY_BILSTM_PATH),
    ("06", BERT_MODEL_DIR / "config.json"),
    ("07", SAFETY_ANN_PATH),
    ("07", SAFETY_SCALER_PATH),
]

missing = 0
for nb, path in CHECKS:
    ok = path.exists()
    missing += not ok
    print(f"[{'OK' if ok else 'MISSING'}] notebook {nb} -> {path.name}")

print("\nAll models present." if not missing else f"\n{missing} missing - run the notebooks marked MISSING.")