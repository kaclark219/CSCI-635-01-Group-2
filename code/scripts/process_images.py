from pathlib import Path
from PIL import Image, ImageEnhance
import random, math, shutil
import cv2
import numpy as np

# define path to data
SPLIT_DIR = Path("data/split")
OUT_ROOT  = Path("data/processed")

# resize images to 256x256
IMAGE_SIZE = (256, 256)
EXTS = (".jpg", ".jpeg", ".png")

# for balancing training data
BAL_ROOT = Path("data/processed_balanced")
TARGET_PER_CLASS = 3000
SEED = 635

# goes through all images in a directory
def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            return_path = p
            yield return_path

def random_affine(img: Image.Image) -> Image.Image:
    '''
    apply random geometric transformations to create extra data for training
    '''
    arr = np.array(img)
    h, w = arr.shape[:2]
    angle = random.uniform(-12, 12)
    tx = random.uniform(-0.08, 0.08) * w
    ty = random.uniform(-0.08, 0.08) * h
    shear = random.uniform(-8, 8)

    M_rot = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty
    sh = math.tan(math.radians(shear))
    M_shear = np.array([[1, sh, 0], [0,  1, 0]], dtype=np.float32)
    M = M_shear @ np.vstack([M_rot, [0,0,1]])
    M = M[:2, :]

    # to prevent icky borders
    warped = cv2.warpAffine(
        arr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    if random.random() < 0.5:
        warped = cv2.flip(warped, 1)

    return Image.fromarray(warped)


def random_color_jitter(img: Image.Image) -> Image.Image:
    '''
    apply random color jitter to create extra data for training
    '''
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
    return img

def augment_once(img: Image.Image) -> Image.Image:
    '''
    apply above augmentations
    '''
    out = random_affine(img)
    out = random_color_jitter(out)
    if out.size != IMAGE_SIZE:
        out = out.resize(IMAGE_SIZE, Image.BILINEAR)
    return out


def resize_images():
    """
    resize images to a standard size and organize into train/val/test folders
    """
    if not SPLIT_DIR.exists():
        raise SystemExit(f"Split folder not found. Make sure to follow README directions: {SPLIT_DIR.resolve()}")

    count = 0
    for split in ("train", "validate", "test"):
        src_split = SPLIT_DIR / split
        if not src_split.exists():
            print(f"Data was not split correctly: {src_split}")
            continue

        for cls_dir in sorted([d for d in src_split.iterdir() if d.is_dir()], key=lambda x: x.name.lower()):
            dst_cls = OUT_ROOT / split / cls_dir.name
            dst_cls.mkdir(parents=True, exist_ok=True)

            for img_path in cls_dir.rglob("*"):
                if not (img_path.is_file() and img_path.suffix.lower() in EXTS):
                    continue
                out_path = dst_cls / img_path.name
                try:
                    with Image.open(img_path) as im:
                        im = im.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
                        im.save(out_path)
                    count += 1
                    if count % 1000 == 0:
                        print(f"Resized {count} images...")
                except Exception as e:
                    print(f"Image skipped {img_path}: {e}")

    print("Done!")

def balance_train_split():
    """
    creates a balanced version of the training set by:
    1. undersampling classes with more than TARGET_PER_CLASS images
    2. augmenting classes with fewer than TARGET_PER_CLASS images
    """
    random.seed(SEED)
    # prep directories
    src_train = OUT_ROOT / "train"
    if not src_train.exists():
        raise SystemExit(f"Processed train split not found: {src_train.resolve()}")
    dst_train = BAL_ROOT / "train"
    class_dirs = sorted([d for d in src_train.iterdir() if d.is_dir()], key=lambda x: x.name.lower())
    if not class_dirs:
        raise SystemExit(f"No class folders in {src_train}")

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        out_cls = dst_train / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        images = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
        n = len(images)
        # undersample
        if n >= TARGET_PER_CLASS:
            chosen = random.sample(images, TARGET_PER_CLASS)
            for i, src in enumerate(chosen):
                dst = out_cls / f"{cls_name}_{i:05d}.jpg"
                shutil.copy2(src, dst)
            print(f"Kept {TARGET_PER_CLASS} (undersampled from {n})")
        # augment
        else:
            for i, src in enumerate(images):
                dst = out_cls / f"{cls_name}_{i:05d}.jpg"
                shutil.copy2(src, dst)
            needed = TARGET_PER_CLASS - n
            for k in range(needed):
                src = random.choice(images)
                with Image.open(src) as im:
                    base = im.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
                aug = augment_once(base)
                dst = out_cls / f"{cls_name}_aug_{k:05d}.jpg"
                aug.save(dst, format="JPEG", quality=92, subsampling=1)
            print(f"Created {needed} augmented to hit total {TARGET_PER_CLASS}")

if __name__ == "__main__":
    # resize_images()
    balance_train_split()