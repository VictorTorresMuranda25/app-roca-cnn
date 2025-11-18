import json
from pathlib import Path
import cv2 as cv

# Rutas fuente (ajústalas si cambiaste)
SRC_ROOT = Path(r"C:\Users\victo\OneDrive\Documentos\Preprocesador imagenes rocas\dataset_manual")
IMAGES_DIR = SRC_ROOT / "images"
ANN_DIR    = SRC_ROOT / "annotations"

# Salida YOLO
OUT_ROOT   = SRC_ROOT / "yolo_seg"
OUT_IMG    = OUT_ROOT / "images"
OUT_LBL    = OUT_ROOT / "labels"

EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def ensure_dirs():
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

def load_json_ann(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instances = data.get("instances", [])
    return instances

def write_yolo_label(img_path: Path, instances):
    # Carga para conocer H,W (necesario para normalizar)
    img = cv.imread(str(img_path))
    if img is None:
        print(f"[WARN] No se pudo leer {img_path.name}, se omite.")
        return
    h, w = img.shape[:2]

    # Archivo de salida
    lbl_path = OUT_LBL / (img_path.stem + ".txt")

    # Si no hay instancias -> archivo vacío (negativa)
    if not instances:
        lbl_path.write_text("", encoding="utf-8")
        return

    # Cada instancia = línea YOLO: "0 x1 y1 x2 y2 ... xN yN"
    lines = []
    for inst in instances:
        poly = inst.get("polygon", [])
        if len(poly) < 3:
            continue
        coords = []
        for pt in poly:
            x = float(pt["x"]) / w
            y = float(pt["y"]) / h
            # clamp por seguridad
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            coords.extend([x, y])
        if len(coords) >= 6:
            line = "0 " + " ".join(f"{v:.6f}" for v in coords)
            lines.append(line)

    lbl_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    ensure_dirs()
    images = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in EXTS])
    if not images:
        print(f"[AVISO] No hay imágenes en {IMAGES_DIR}")
        return

    for img_path in images:
        # Copiamos imagen a la carpeta YOLO (puedes omitir si ya apuntas a IMAGES_DIR original)
        dst_img = OUT_IMG / img_path.name
        if dst_img.resolve() != img_path.resolve():
            # ahorra espacio duro: solo copia si cambias de carpeta, si quieres puedes usar hardlink
            import shutil
            shutil.copy2(img_path, dst_img)
        else:
            dst_img = img_path

        json_path = ANN_DIR / f"{img_path.stem}.json"
        if json_path.exists():
            instances = load_json_ann(json_path)
        else:
            instances = []  # negativa

        write_yolo_label(dst_img, instances)

    print(f"[OK] Conversión terminada.\nImágenes: {OUT_IMG}\nEtiquetas: {OUT_LBL}")

if __name__ == "__main__":
    main()
