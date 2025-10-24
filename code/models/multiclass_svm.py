from pathlib import Path
from sklearn import svm

IMG_DIR = Path("../../data/processed")
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR   = IMG_DIR / "validate"
TEST_DIR  = IMG_DIR / "test"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

RESULTS_DIR = Path("/results")
RESULTS_PATH = RESULTS_DIR / "multiclass_svm_results.csv"

TARGET_STYLES = [
    "Abstract_Expressionism",
    "Baroque",
    "Cubism",
    "High_Renaissance",
    "Impressionism",
    "Pop_Art",
    "Realism",
]