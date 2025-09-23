from pathlib import Path
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# define path to data
IMG_DIR = Path("data/processed")
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR = IMG_DIR / "validate"
TEST_DIR = IMG_DIR / "test"

# define styles for the model to target
TARGET_STYLES = [
    "Abstract_Expressionism",
    "Baroque",
    "Cubism",
    "High_Renaissance",
    "Impressionism",
    "Pop_Art",
    "Realism"
]

# define hyperparameters
K = 10

def single_kNN():
    pass

def bagged_kNN():
    single_kNN()
    pass

if __name__ == "__main__":
    bagged_kNN()