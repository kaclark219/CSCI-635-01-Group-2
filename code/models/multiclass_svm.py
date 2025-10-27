from pathlib import Path
from typing import Tuple, List
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, matthews_corrcoef, make_scorer)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from joblib import Memory, parallel_backend

# config
QUICK_MODE = True
THREADS = 4
CV_FOLDS_QUICK = 2
CV_FOLDS_FULL = 3

IMG_DIR = Path("../../data/processed")
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR   = IMG_DIR / "validate"
TEST_DIR  = IMG_DIR / "test"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

RESULTS_DIR = Path("/results")
RESULTS_PATH = RESULTS_DIR / "svm_results.csv"
GRID_CSV_PATH = RESULTS_DIR / "svm_gridsearch_mcc.csv"

TARGET_STYLES = [
    "Abstract_Expressionism",
    "Baroque",
    "Cubism",
    "High_Renaissance",
    "Impressionism",
    "Pop_Art",
    "Realism",
]

RANDOM_STATE = 635

if QUICK_MODE:
    IMAGE_SIZE = (96, 96)
    USE_PCA = True
    PCA_COMPONENTS = 128
    CV_FOLDS = CV_FOLDS_QUICK
else:
    IMAGE_SIZE = (128, 128)
    USE_PCA = True
    PCA_COMPONENTS = 256
    CV_FOLDS = CV_FOLDS_FULL

# pipeline cache
PIPE_CACHE_DIR = Path("pipeline_cache")
PIPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PIPE_MEMORY = Memory(location=str(PIPE_CACHE_DIR), verbose=0)

def iter_images(split_dir: Path, classes: List[str]) -> List[Tuple[Path, str]]:
    items = []
    for cls in classes:
        cdir = split_dir / cls
        for p in cdir.rglob("*"):
            if p.suffix.lower() in EXTS and p.is_file():
                items.append((p, cls))
    return items

def load_split(split_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    pairs = iter_images(split_dir, TARGET_STYLES)
    X, y = [], []
    for p, label in pairs:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.size != IMAGE_SIZE:
                    im = im.resize(IMAGE_SIZE, Image.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8).reshape(-1)
                X.append(arr)
                y.append(label)
        except Exception as e:
            print(f"Error loading image {p}: {e}")
    X = (np.stack(X).astype(np.float16)) / np.float16(255.0)
    y = np.array(y)
    return X, y

# log results to csv file
def append_result(model_name: str, kernel: str, C: float, gamma, val_acc: float, test_acc: float, note: str = "svm"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    header_needed = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,model,kernel,C,gamma,val_acc,test_acc,note,quick_mode,image_size,pca_components,cv_folds,threads\n")
        ts = datetime.now().isoformat(timespec="seconds")
        gval = "" if gamma is None else gamma
        line = f"{ts},{model_name},{kernel},{C},{gval},{val_acc:.6f},{test_acc:.6f},{note},{QUICK_MODE},{IMAGE_SIZE},{PCA_COMPONENTS if USE_PCA else ''},{CV_FOLDS},{THREADS}\n"
        f.write(line)

# build svm model pipeline
def make_pipeline() -> Pipeline:
    steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]
    if USE_PCA:
        steps.append(("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE, svd_solver="randomized")))
    steps.append(("clf", LinearSVC()))
    return Pipeline(steps, memory=PIPE_MEMORY)

# actual svm implementation with grid search
def run_svm_pipeline():
    t0 = time.time()
    Xtr, ytr = load_split(TRAIN_DIR)
    Xva, yva = load_split(VAL_DIR)
    Xte, yte = load_split(TEST_DIR)
    pipe = make_pipeline()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mcc_scorer = make_scorer(matthews_corrcoef)

    linear_est = LinearSVC(
        class_weight="balanced",
        dual=False,
        max_iter=8000,
        tol=1e-3,
        random_state=RANDOM_STATE
    )
    linear_grid = {
        "clf": [linear_est],
        "clf__C": [0.3, 1, 3] if QUICK_MODE else [0.1, 0.3, 1, 3, 10],
    }

    rbf_est = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        cache_size=1000,
        tol=1e-2,
        max_iter=-1,
        random_state=RANDOM_STATE,
    )
    rbf_grid = {
        "clf": [rbf_est],
        "clf__C": [1, 3] if QUICK_MODE else [0.3, 1, 3, 10],
        "clf__gamma": ["scale", 1e-3] if QUICK_MODE else ["scale", 1e-3, 1e-4],
    }

    PARAM_GRID = [linear_grid, rbf_grid]
    if QUICK_MODE:
        search = HalvingGridSearchCV(
            estimator=pipe,
            param_grid=PARAM_GRID,
            scoring=mcc_scorer,
            cv=cv,
            factor=3,
            resource="n_samples",
            aggressive_elimination=True,
            n_jobs=1,
            return_train_score=False,
            verbose=1,
        )
    else:
        search = GridSearchCV(
            estimator=pipe,
            param_grid=PARAM_GRID,
            scoring=mcc_scorer,
            cv=cv,
            n_jobs=1,
            pre_dispatch=1,
            return_train_score=True,
            verbose=1,
        )

    # threads to share memory use, attempt to avoid crashing due to memory spikes
    with parallel_backend("threading", n_jobs=THREADS):
        search.fit(Xtr, ytr)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(GRID_CSV_PATH, index=False)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV MCC: {search.best_score_:.4f}")

    best_pipe: Pipeline = search.best_estimator_
    best_model = best_pipe.named_steps["clf"]

    # figure out which model was best & pullparams
    if isinstance(best_model, LinearSVC):
        model_name, kernel = "LinearSVC", "linear"
        C = best_model.C
        gamma = None
    else:
        model_name, kernel = "SVC", "rbf"
        C = best_model.C
        gamma = best_model.gamma

    # validation
    yva_pred = best_pipe.predict(Xva)
    val_acc = accuracy_score(yva, yva_pred)
    print(f"\nValidation accuracy: {val_acc:.4f}")
    print(classification_report(yva, yva_pred, target_names=TARGET_STYLES, digits=4))
    print("Val Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yva, yva_pred, labels=TARGET_STYLES))

    # refit using the best params on train & val
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])
    final_pipe = make_pipeline()
    final_pipe.set_params(**search.best_params_)
    with parallel_backend("threading", n_jobs=THREADS):
        final_pipe.fit(Xtrva, ytrva)

    # testing
    yte_pred = final_pipe.predict(Xte)
    test_acc = accuracy_score(yte, yte_pred)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(yte, yte_pred, target_names=TARGET_STYLES, digits=4))
    print("Test Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, yte_pred, labels=TARGET_STYLES))

    append_result(model_name, kernel, C, gamma, val_acc, test_acc, note="svm/linear_or_rbf")

if __name__ == "__main__":
    run_svm_pipeline()