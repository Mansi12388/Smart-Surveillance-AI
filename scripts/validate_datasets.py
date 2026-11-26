"""Validate whichever datasets exist. Run from project root: python scripts/validate_datasets.py"""
from pathlib import Path
import cv2, pandas as pd

D = Path(__file__).resolve().parents[1] / "datasets"
ok = True

def report(name, df, label_col):
    print(f"\n== {name}: {len(df)} rows ==")
    print(df[label_col].value_counts().to_string())

# Text
p = D / "incident_text" / "incidents.csv"
if p.exists():
    df = pd.read_csv(p)
    assert list(df.columns) == ["text", "label"], "incidents.csv must have columns text,label"
    assert df.isna().sum().sum() == 0 and not df.duplicated().any(), "missing/duplicate text rows"
    report("incident_text (SYNTHETIC)", df, "label")
else:
    print("incidents.csv missing"); ok = False

# Structured
p = D / "safety_events" / "safety_events.csv"
if p.exists():
    df = pd.read_csv(p)
    need = {"duration_seconds", "people_count", "mask_issue_count", "motion_change_count", "event_category"}
    assert need <= set(df.columns) and df.isna().sum().sum() == 0
    report("safety_events (SYNTHETIC)", df, "event_category")
else:
    print("safety_events.csv missing"); ok = False

# Face mask
p = D / "face_mask" / "face_mask_labels.csv"
if p.exists():
    df = pd.read_csv(p)
    bad = [x for x in df["path"] if cv2.imread(str(D / "face_mask" / x)) is None]
    leak = df.groupby("source_image")["split"].nunique().gt(1).sum()
    report("face_mask", df, "label"); print("unreadable:", len(bad), "| source images in >1 split:", leak)
    ok &= not bad and leak == 0
else:
    print("face_mask_labels.csv not created yet (run prepare_face_mask.py after downloading)")

# Activity
p = D / "human_activity" / "activity_labels.csv"
if p.exists():
    df = pd.read_csv(p)
    bad = [x for x in df["video_path"] if not cv2.VideoCapture(str(D / "human_activity" / x)).isOpened()]
    leak = df.groupby(["label", "source_group"])["split"].nunique().gt(1).sum()
    report("human_activity", df, "label"); print("unreadable:", len(bad), "| source groups in >1 split:", leak)
    ok &= not bad and leak == 0
else:
    print("activity_labels.csv not created yet (run prepare_hmdb51.py after downloading)")

print("\nVALIDATION", "PASSED" if ok else "HAS ISSUES")