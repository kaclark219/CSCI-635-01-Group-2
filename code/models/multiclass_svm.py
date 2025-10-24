from pathlib import Path
from typing import Tuple, List
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef, make_scorer


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
IMAGE_SIZE = (128, 128)

USE_PCA = True
PCA_COMPONENTS = 256

# param grid for svm
PARAM_GRID = [
    {"svc__kernel": ["linear"], "svc__C": [0.1, 0.3, 1, 3, 10]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 0.3, 1, 3, 10], "svc__gamma": ["scale", 1e-3, 1e-4]},
]

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
    X = np.stack(X).astype(np.float32) / 255.0
    y = np.array(y)
    return X, y

# log results to csv file
def append_result(kernel: str, C: float, gamma, val_acc: float, test_acc: float, note: str = "svm"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    header_needed = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,kernel,C,gamma,val_acc,test_acc,note\n")
        ts = datetime.now().isoformat(timespec="seconds")
        gval = "" if gamma is None else gamma
        f.write(f"{ts},{kernel},{C},{gval},{val_acc:.6f},{test_acc:.6f},{note}\n")

# build svm model pipeline
def make_pipeline() -> Pipeline:
    steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]
    if USE_PCA:
        steps.append(("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE, svd_solver="randomized")))
    steps.append(("svc", SVC(decision_function_shape="ovo", class_weight="balanced", random_state=RANDOM_STATE)))
    return Pipeline(steps)

# actual svm implementation with grid search
def run_svm_pipeline():
    Xtr, ytr = load_split(TRAIN_DIR)
    Xva, yva = load_split(VAL_DIR)
    Xte, yte = load_split(TEST_DIR)

    # gridsearch for parameters
    pipe = make_pipeline()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    mcc_scorer = make_scorer(matthews_corrcoef)

    gs = GridSearchCV(estimator=pipe, param_grid=PARAM_GRID, scoring=mcc_scorer, cv=cv, n_jobs=-1, return_train_score=True)
    gs.fit(Xtr, ytr)

    # save results from gridsearch
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gs.cv_results_).to_csv(GRID_CSV_PATH, index=False)
    print(f"\nGridSearch complete. Best params: {gs.best_params_}")
    print(f"Best CV MCC: {gs.best_score_:.4f}")

    # validation
    best_pipe: Pipeline = gs.best_estimator_
    yva_pred = best_pipe.predict(Xva)
    val_acc = accuracy_score(yva, yva_pred)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(classification_report(yva, yva_pred, target_names=TARGET_STYLES, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yva, yva_pred, labels=TARGET_STYLES))

    # refit using the best params on train & val
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])
    final_pipe = make_pipeline()
    final_pipe.set_params(**gs.best_params_)
    final_pipe.fit(Xtrva, ytrva)

    # testing
    yte_pred = final_pipe.predict(Xte)
    test_acc = accuracy_score(yte, yte_pred)
    print(f"Test accuracy: {test_acc:.4f}")
    print(classification_report(yte, yte_pred, target_names=TARGET_STYLES, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(yte, yte_pred, labels=TARGET_STYLES))

    # save results of model
    kernel = gs.best_params_.get("svc__kernel")
    C = gs.best_params_.get("svc__C")
    gamma = gs.best_params_.get("svc__gamma") if kernel == "rbf" else None
    append_result(kernel, C, gamma, val_acc, test_acc, note="svm")

    # actually save them model
    out_dir = Path("models"); out_dir.mkdir(parents=True, exist_ok=True)
    from joblib import dump # FIX: issues with array memory ... not running all the way through
    dump({
        "model": final_pipe,
        "best_params": gs.best_params_,
        "best_cv_mcc": gs.best_score_,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "classes": TARGET_STYLES,
        "use_pca": USE_PCA,
        "pca_components": PCA_COMPONENTS if USE_PCA else None,
        "image_size": IMAGE_SIZE,
    }, out_dir / "best_svm.joblib")
    print(f"Saved model to { (out_dir / 'best_svm.joblib').resolve() }")

if __name__ == "__main__":
    run_svm_pipeline()