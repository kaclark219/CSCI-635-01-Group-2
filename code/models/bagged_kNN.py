from pathlib import Path
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime

# too many features .. need to use PCA
USE_PCA = True
PCA_COMPONENTS = 256

# paths
IMG_DIR = Path("data/processed")
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR   = IMG_DIR / "validate"
TEST_DIR  = IMG_DIR / "test"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

RESULTS_DIR = Path("models/results")
RESULTS_PATH = RESULTS_DIR / "bagged_knn_results.csv"

TARGET_STYLES = [
    "Abstract_Expressionism",
    "Baroque",
    "Cubism",
    "High_Renaissance",
    "Impressionism",
    "Pop_Art",
    "Realism"
]

K = 10

def load_split(split_dir: Path):
    class_to_idx = {c: i for i, c in enumerate(TARGET_STYLES)}
    X, y = [], []
    total = 0
    for cls in TARGET_STYLES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        files = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        for i, p in enumerate(files, 1):
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    arr = np.asarray(im, dtype=np.uint8)
                    X.append(arr.reshape(-1))
                    y.append(class_to_idx[cls])
            except Exception as e:
                print(f"skip {p}: {e}", flush=True)
            if i % 500 == 0:
                print(f"...{i}/{len(files)} processed for {cls}", flush=True)
        total += len(files)
    if not X:
        raise SystemExit(f"ERROR no images found under {split_dir}")
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def append_result(k_val: int, val_acc: float, test_acc: float):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    header_needed = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,k,val_acc,test_acc\n")
        ts = datetime.now().isoformat(timespec="seconds")
        f.write(f"{ts},{k_val},{val_acc:.6f},{test_acc:.6f}\n")
    print(f"results logged to {RESULTS_PATH}", flush=True)

def single_kNN(return_data: bool = False):
    Xtr, ytr = load_split(TRAIN_DIR)
    Xva, yva = load_split(VAL_DIR)
    Xte, yte = load_split(TEST_DIR)

    if USE_PCA:
        Xtr /= 255.0
        Xva /= 255.0
        Xte /= 255.0

        pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xva = pca.transform(Xva)
        Xte = pca.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean", n_jobs=-1)
    knn.fit(Xtr, ytr)

    yva_pred = knn.predict(Xva)
    val_acc  = accuracy_score(yva, yva_pred)
    print(classification_report(yva, yva_pred, target_names=TARGET_STYLES, digits=4))

    yte_pred = knn.predict(Xte)
    test_acc = accuracy_score(yte, yte_pred)
    print(classification_report(yte, yte_pred, target_names=TARGET_STYLES, digits=4))

    if return_data:
        return Xtr, ytr, Xva, yva, Xte, yte

def bagged_kNN():
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

    bag.fit(Xtr, ytr)

    yva_pred = bag.predict(Xva)
    val_acc  = accuracy_score(yva, yva_pred)
    print(classification_report(yva, yva_pred, target_names=TARGET_STYLES, digits=4))

    yte_pred = bag.predict(Xte)
    test_acc = accuracy_score(yte, yte_pred)
    print(classification_report(yte, yte_pred, target_names=TARGET_STYLES, digits=4))

    append_result(K, val_acc, test_acc)

if __name__ == "__main__":
    bagged_kNN()