import random, shutil
from pathlib import Path
import yaml

YOLO_ROOT = Path(r"C:\Users\victo\OneDrive\Documentos\Preprocesador imagenes rocas\dataset_manual\yolo_seg")
IMAGES_DIR = YOLO_ROOT / "images"
LABELS_DIR = YOLO_ROOT / "labels"

# Estructura dataset final
DST_ROOT = YOLO_ROOT / "dataset_yolo"
TR_IMG = DST_ROOT / "train" / "images"
TR_LBL = DST_ROOT / "train" / "labels"
VL_IMG = DST_ROOT / "val"   / "images"
VL_LBL = DST_ROOT / "val"   / "labels"

EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def ensure_dirs():
    for p in [TR_IMG, TR_LBL, VL_IMG, VL_LBL]:
        p.mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    imgs = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in EXTS])
    if not imgs:
        print(f"[AVISO] No hay imágenes en {IMAGES_DIR}")
        return
    random.seed(42)
    random.shuffle(imgs)
    n = len(imgs)
    n_val = max(1, int(0.2*n))
    val_set = set(imgs[:n_val])
    train_set = set(imgs[n_val:])

    for p in train_set:
        lbl = LABELS_DIR / (p.stem + ".txt")
        shutil.copy2(p, TR_IMG / p.name)
        shutil.copy2(lbl, TR_LBL / lbl.name)

    for p in val_set:
        lbl = LABELS_DIR / (p.stem + ".txt")
        shutil.copy2(p, VL_IMG / p.name)
        shutil.copy2(lbl, VL_LBL / lbl.name)

    # data.yaml
    data = {
        "path": str(DST_ROOT),
        "train": "train/images",
        "val":   "val/images",
        "names": {0: "rock"}
    }
    with open(DST_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Split listo en: {DST_ROOT}\nTrain: {len(train_set)}  Val: {len(val_set)}")

if __name__ == "__main__":
    main()
