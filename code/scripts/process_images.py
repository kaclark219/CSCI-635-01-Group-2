from pathlib import Path
from PIL import Image

# define path to data
SPLIT_DIR = Path("data/split")
OUT_ROOT  = Path("data/processed")

# resize images to 224x224 (standard)
IMAGE_SIZE = (224, 224)
EXTS = (".jpg", ".jpeg", ".png")

# goes through all images in a directory
def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            return_path = p
            yield return_path

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

if __name__ == "__main__":
    resize_images()