from pathlib import Path

DEFAULT_RANDOM_STATE = 49
DEFAULT_TEST_SIZE = 0.25
DEFAULT_Z_THRESHOLD = 3.0

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
# Kaggle-style training file (same schema as notebook); keep under data/ locally (gitignored).
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "Data_Train.xlsx"
