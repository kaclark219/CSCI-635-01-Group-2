from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import random

# define path to raw data & paths to future output folders
RAW_DIR = Path("data/raw")
OUT_ROOT = Path("data/split")
TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR = OUT_ROOT / "validate"
TEST_DIR = OUT_ROOT / "test"

# split raw data into folders for training, validation, and testing .. keeping them in the correct subfolders for each class
TRAIN_PCT, VAL_PCT, TEST_PCT = 0.6, 0.2, 0.2 # ratios for splitting data
SEED = 635
EXTS = (".jpg", ".jpeg", ".png") # looks like all jpgs, but just in case

# utility functions
def list_images(d: Path):
    return [p for p in d.rglob("*") if p.suffix.lower() in EXTS]

def safe_copy(files, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        target = dest_dir / src.name
        if target.exists():
            stem, suf = target.stem, target.suffix
            i = 1
            while (dest_dir / f"{stem}_{i}{suf}").exists():
                i += 1
            target = dest_dir / f"{stem}_{i}{suf}"
        shutil.copy2(src, target)

def main():
    if not RAW_DIR.exists(): # make sure that you created a raw folder & put images in there
        raise SystemExit(f"Raw dir not found: {RAW_DIR.resolve()}")

    # find the subclass folders needed for art style classification
    class_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if not class_dirs:
        raise SystemExit(f"No class folders found under {RAW_DIR}")

    # create the output folders if they don't already exist
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # for each class, split the images into train/val/test sets & copy to the output folders
    rng = random.Random(SEED)
    print(f"Found {len(class_dirs)} classes. Splitting {int(TRAIN_PCT*100)}/{int(VAL_PCT*100)}/{int(TEST_PCT*100)} ...\n") # just a checkpoint

    for cls_dir in class_dirs:
        cls = cls_dir.name
        imgs = list_images(cls_dir)

        if len(imgs) < 3: # need at least 3 images to split into the three folders
            print(f"Skipping {cls}: not enough images ({len(imgs)})")
            continue
        train_files, temp_files = train_test_split(
            imgs, test_size=(1 - TRAIN_PCT), random_state=SEED, shuffle=True
        )
        val_ratio_in_temp = VAL_PCT / (VAL_PCT + TEST_PCT) if (VAL_PCT + TEST_PCT) > 0 else 0.0
        val_files, test_files = train_test_split(
            temp_files, test_size=(1 - val_ratio_in_temp), random_state=SEED, shuffle=True
        )

        # copy into folders for training, validation, testing
        safe_copy(train_files, TRAIN_DIR / cls)
        safe_copy(val_files,   VAL_DIR   / cls)
        safe_copy(test_files,  TEST_DIR  / cls)

    print("Done!")

if __name__ == "__main__":
    main()