from pathlib import Path
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# define path to data
IMG_DIR = Path("data/processed")
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR = IMG_DIR / "validate"
TEST_DIR = IMG_DIR / "test"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

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

# load pre-processed/split data into np arrays
def load_split(split_dir: Path):
        X, y = [], []
        class_to_idx = {c: i for i, c in enumerate(TARGET_STYLES)}
        for cls in TARGET_STYLES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"Missing class dir: {cls_dir}")
                continue
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in EXTS:
                    try:
                        with Image.open(p) as im:
                            im = im.convert("RGB")
                            arr = np.asarray(im, dtype=np.uint8)
                            X.append(arr.reshape(-1))
                            y.append(class_to_idx[cls])
                    except Exception as e:
                        print("skipped", p, e)
        if not X:
            raise SystemExit(f"No images found under {split_dir}")
        return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)

def single_kNN(return_data: bool = False):
    '''
    single kNN as starting point for bagging method
    '''
    Xtr, ytr = load_split(TRAIN_DIR)
    Xva, yva = load_split(VAL_DIR)
    Xte, yte = load_split(TEST_DIR)

    # training
    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean", n_jobs=-1)
    knn.fit(Xtr, ytr)

    # validation
    yva_pred = knn.predict(Xva)
    print(f"[single kNN] val acc: {accuracy_score(yva, yva_pred):.4f}")
    print(classification_report(yva, yva_pred, target_names=TARGET_STYLES, digits=4))

    # testing
    yte_pred = knn.predict(Xte)
    print(f"[single kNN] test acc: {accuracy_score(yte, yte_pred):.4f}")
    print(classification_report(yte, yte_pred, target_names=TARGET_STYLES, digits=4))

    if return_data:
        return Xtr, ytr, Xva, yva, Xte, yte

def bagged_kNN():
    '''
    bagged knn classifier
    '''
    Xtr, ytr, Xva, yva, Xte, yte = single_kNN(return_data=True)

    base = KNeighborsClassifier(n_neighbors=K, metric="euclidean", n_jobs=-1)
    bag  = BaggingClassifier(
        estimator=base,
        n_estimators=15,
        max_samples=0.9,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )

    # training
    print("training bagged k-NN…")
    bag.fit(Xtr, ytr)

    # validation
    yva_pred = bag.predict(Xva)
    print(f"[bagged kNN] val acc: {accuracy_score(yva, yva_pred):.4f}")
    print(classification_report(yva, yva_pred, target_names=TARGET_STYLES, digits=4))

    # testing
    yte_pred = bag.predict(Xte)
    print(f"[bagged kNN] test acc: {accuracy_score(yte, yte_pred):.4f}")
    print(classification_report(yte, yte_pred, target_names=TARGET_STYLES, digits=4))

if __name__ == "__main__":
    bagged_kNN()