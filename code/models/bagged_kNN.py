from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, matthews_corrcoef, make_scorer)
import cv2
from skimage.feature import local_binary_pattern
from datetime import datetime


USE_PCA = True # to reduce feature dimensions before knn
PCA_COMPONENTS = 256

# grid for testing k in knn
K_GRID = [1, 3, 5, 7, 9, 11]

IMG_DIR = Path("../../data/processed")
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR   = IMG_DIR / "validate"
TEST_DIR  = IMG_DIR / "test"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

RESULTS_DIR = Path("/results")
RESULTS_PATH = RESULTS_DIR / "bagged_knn_results.csv"

TARGET_STYLES = [
    "Abstract_Expressionism",
    "Baroque",
    "Cubism",
    "High_Renaissance",
    "Impressionism",
    "Pop_Art",
    "Realism",
]

# feature extracting functions
def hsv_hists(img, h_bins=16, s_bins=8, v_bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv],[0],None,[h_bins],[0,180]).flatten()
    s_hist = cv2.calcHist([hsv],[1],None,[s_bins],[0,256]).flatten()
    v_hist = cv2.calcHist([hsv],[2],None,[v_bins],[0,256]).flatten()
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist

def hsv_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    means = hsv.mean(axis=(0,1))
    stds  = hsv.std(axis=(0,1))
    return np.concatenate([means, stds]).astype(np.float32)

def lbp_hist(img, P=8, R=1):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist

def edge_density(img, low=100, high=200):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return np.array([edges.mean()], dtype=np.float32)

def hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = cv2.moments(gray)
    hu = cv2.HuMoments(m).flatten()
    hu = np.sign(hu) * np.log1p(np.abs(hu))
    return hu.astype(np.float32)

def extract_features_from_image(img_path, size=(128,128)):
    # ensure 128x128 RGB
    with Image.open(img_path) as im:
        im = im.convert("RGB").resize(size, Image.BICUBIC)
    img = np.array(im)

    feats = [
        hsv_hists(img),
        hsv_stats(img),
        lbp_hist(img),
        edge_density(img),
        hu_moments(img),
    ]
    return np.concatenate(feats).astype(np.float32)


# load data from processed splits
def load_split(split_dir: Path):
    class_to_idx = {c: i for i, c in enumerate(TARGET_STYLES)}
    X, y = [], []
    for cls in TARGET_STYLES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        files = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        for i, p in enumerate(files, 1):
            try:
                feat = extract_features_from_image(p)
                X.append(feat)
                y.append(class_to_idx[cls])
            except Exception as e:
                print(f"skip {p}: {e}", flush=True)
            if i % 500 == 0:
                print(f"...{i}/{len(files)} processed for {cls}", flush=True)
    if not X:
        raise SystemExit(f"ERROR no images found under {split_dir}")
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

# log results to csv file
def append_result(k_val: int, val_acc: float, test_acc: float, note: str = "bagged_knn"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    header_needed = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,k,val_acc,test_acc,note\n")
        ts = datetime.now().isoformat(timespec="seconds")
        f.write(f"{ts},{k_val},{val_acc:.6f},{test_acc:.6f},{note}\n")


# actual knn implementation with bagging + use of grid search
def run_knn_pipeline():
    Xtr, ytr = load_split(TRAIN_DIR)
    Xva, yva = load_split(VAL_DIR)
    Xte, yte = load_split(TEST_DIR)

    # scale features
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # reduce dimensions
    if USE_PCA:
        m, d = Xtr_s.shape
        max_allowed = max(1, min(m - 1, d))
        n_comp = min(PCA_COMPONENTS, max_allowed)
        pca = PCA(n_components=n_comp, svd_solver="auto", random_state=42)
        Xtr_s = pca.fit_transform(Xtr_s)
        Xva_s = pca.transform(Xva_s)
        Xte_s = pca.transform(Xte_s)

    # gridsearch for best k in knn w mcc scoring
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=35)
    mcc_scorer = make_scorer(matthews_corrcoef)

    knn = KNeighborsClassifier(metric="euclidean", n_jobs=-1)
    knn_param_grid = {"n_neighbors": K_GRID}
    gs = GridSearchCV(estimator=knn, param_grid=knn_param_grid, cv=cv, scoring=mcc_scorer, return_train_score=True, n_jobs=-1)
    gs.fit(Xtr_s, ytr)
    knn_df = pd.DataFrame(gs.cv_results_)

    best_k = int(gs.best_params_["n_neighbors"])
    best_mcc = gs.best_score_

    # try regular knn with best k for comparison
    knn_best = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
    knn_best.fit(Xtr_s, ytr)
    yva_pred = knn_best.predict(Xva_s)
    val_acc = accuracy_score(yva, yva_pred)
    print(classification_report(yva, yva_pred, target_names=TARGET_STYLES, digits=4))
    yte_pred = knn_best.predict(Xte_s)
    test_acc = accuracy_score(yte, yte_pred)
    print(classification_report(yte, yte_pred, target_names=TARGET_STYLES, digits=4))

    # bagged knn with best k
    base = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
    bag = BaggingClassifier(estimator=base, n_estimators=15, max_samples=0.9, bootstrap=True, n_jobs=-1, random_state=35)
    bag.fit(Xtr_s, ytr)
    yva_pred_bag = bag.predict(Xva_s)
    val_acc_bag = accuracy_score(yva, yva_pred_bag)
    print(classification_report(yva, yva_pred_bag, target_names=TARGET_STYLES, digits=4))

    # evaluation using test data
    Xtrva_s = np.vstack([Xtr_s, Xva_s])
    ytrva = np.concatenate([ytr, yva])
    bag.fit(Xtrva_s, ytrva)
    yte_pred_bag = bag.predict(Xte_s)
    test_acc_bag = accuracy_score(yte, yte_pred_bag)
    print(classification_report(yte, yte_pred_bag, target_names=TARGET_STYLES, digits=4))

    # save results
    append_result(best_k, val_acc_bag, test_acc_bag, note="bagged_knn")
    append_result(best_k, val_acc, test_acc, note="plain_knn")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    knn_df.to_csv(RESULTS_DIR / "knn_gridsearch_mcc.csv", index=False)

if __name__ == "__main__":
    run_knn_pipeline()