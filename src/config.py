"""
Central configuration for Smart Surveillance AI.
Every other file imports paths and constants from here.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"

# face mask
MASK_DIR = DATASETS_DIR / "face_mask"
MASK_IMAGES_DIR = MASK_DIR / "images"
MASK_TRAIN_DIR = MASK_IMAGES_DIR / "train"
MASK_VAL_DIR = MASK_IMAGES_DIR / "val"
MASK_TEST_DIR = MASK_IMAGES_DIR / "test"
MASK_LABELS_CSV = MASK_DIR / "face_mask_labels.csv"

# human activity
ACTIVITY_DIR = DATASETS_DIR / "human_activity"
ACTIVITY_VIDEOS_DIR = ACTIVITY_DIR / "videos"
ACTIVITY_FRAMES_DIR = ACTIVITY_DIR / "frames"
ACTIVITY_LABELS_CSV = ACTIVITY_DIR / "activity_labels.csv"

# incident text
INCIDENT_CSV = DATASETS_DIR / "incident_text" / "incidents.csv"

# safety events
SAFETY_CSV = DATASETS_DIR / "safety_events" / "safety_events.csv"

# saved models
MASK_CNN_PATH = MODELS_DIR / "mask_cnn.keras"
MASK_MOBILENET_PATH = MODELS_DIR / "mask_mobilenetv2.keras"
ACTIVITY_CNN_LSTM_PATH = MODELS_DIR / "activity_cnn_lstm.keras"
ACTIVITY_BILSTM_PATH = MODELS_DIR / "activity_bilstm.keras"
BERT_MODEL_DIR = MODELS_DIR / "bert_incident"
SAFETY_ANN_PATH = MODELS_DIR / "safety_ann.keras"
SAFETY_SCALER_PATH = MODELS_DIR / "safety_scaler.pkl"

# database
DATABASE_PATH = DATABASE_DIR / "surveillance.db"

# ---------------------------------------------------------------- general
SEED = 42

# ---------------------------------------------------------------- face mask
MASK_IMG_SIZE = (224, 224)
MASK_BATCH_SIZE = 32
# alphabetical order = the order Keras flow_from_directory uses
MASK_CLASSES = ["mask_weared_incorrect", "with_mask", "without_mask"]

# ---------------------------------------------------------------- activity
ACTIVITY_IMG_SIZE = (224, 224)
ACTIVITY_NUM_FRAMES = 16          # frames saved per video
ACTIVITY_WINDOW = 8               # window length for LSTM / BiLSTM samples
ACTIVITY_WINDOW_STRIDE = 2
ACTIVITY_FEATURE_DIM = 1280       # MobileNetV2 global-average-pooled features
ACTIVITY_CLASSES = ["Sitting", "Standing", "Walking"]

# ---------------------------------------------------------------- incident text
INCIDENT_CLASSES = ["Facility Issue", "Fall Report", "Normal Activity", "Rapid Movement"]
BERT_BASE_MODEL = "bert-base-uncased"
BERT_MAX_LEN = 64

# ---------------------------------------------------------------- safety events
SAFETY_FEATURES = ["duration_seconds", "people_count",
                   "mask_issue_count", "motion_change_count"]
SAFETY_TARGET = "event_category"
SAFETY_CLASSES = ["long_group_event_demo", "mask_observation_demo",
                  "movement_observation_demo", "normal_demo"]

# ---------------------------------------------------------------- api
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"


def ensure_dirs() -> None:
    """Create folders that must exist before saving models / the database."""
    MODELS_DIR.mkdir(exist_ok=True)
    DATABASE_DIR.mkdir(exist_ok=True)


ensure_dirs()