import argparse
import json
from pathlib import Path
import cv2 as cv
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# estado global para callbacks
current_points = []   # puntos del polígono en edición
polygons = []         # lista de polígonos confirmados (cada uno: [(x,y), ...])
img_orig = None       # imagen original actual
img_disp = None       # imagen con overlay actual

def draw_overlay():
    """Dibuja polígonos confirmados y el polígono en edición sobre img_disp."""
    global img_disp, img_orig, polygons, current_points
    img_disp = img_orig.copy()

    # polígonos confirmados
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv.fillPoly(img_disp, [pts], color=(0, 180, 0))         # relleno tenue
        cv.polylines(img_disp, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # polígono en edición
    if len(current_points) > 0:
        pts = np.array(current_points, dtype=np.int32)
        cv.polylines(img_disp, [pts], isClosed=False, color=(0, 200, 255), thickness=2)

    # leyenda
    txt = ("LClick: punto | RClick: cerrar polígono | U: undo punto | Z: undo polígono | "
           "C: limpiar en edición | S: guardar | N: guardar+siguiente | B: guardar+anterior | Q/Esc: salir")
    cv.putText(img_disp, txt, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 255), 2, cv.LINE_AA)

def on_mouse(event, x, y, flags, param):
    global current_points, polygons
    if event == cv.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        draw_overlay()
    elif event == cv.EVENT_RBUTTONDOWN:
        # cerrar polígono (si hay ≥3 puntos)
        if len(current_points) >= 3:
            polygons.append(current_points.copy())
            current_points = []
            draw_overlay()

def save_annotation(img_path: Path, out_masks: Path, out_json: Path):
    """Guarda máscara con IDs de instancia y JSON con los vértices."""
    global polygons, img_orig
    h, w = img_orig.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint16)  # 0=fondo, 1..K=instancias

    for idx, poly in enumerate(polygons, start=1):
        pts = np.array(poly, dtype=np.int32)
        cv.fillPoly(mask, [pts], color=idx)

    out_masks.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)

    mask_path = out_masks / f"{img_path.stem}_mask.png"
    json_path = out_json / f"{img_path.stem}.json"

    cv.imwrite(str(mask_path), mask)

    data = {
        "image": img_path.name,
        "height": int(h),
        "width": int(w),
        "instances": [
            {"id": i + 1, "polygon": [{"x": int(x), "y": int(y)} for (x, y) in poly]}
            for i, poly in enumerate(polygons)
        ]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Guardado {mask_path.name} y {json_path.name}")

def load_previous_if_any(img_path: Path, out_json: Path):
    """Si ya existe anotación, la carga para continuar editando."""
    global polygons
    polygons = []
    j = out_json / f"{img_path.stem}.json"
    if j.exists():
        try:
            with open(j, "r", encoding="utf-8") as f:
                data = json.load(f)
            for inst in data.get("instances", []):
                poly = [(int(p["x"]), int(p["y"])) for p in inst.get("polygon", [])]
                if len(poly) >= 3:
                    polygons.append(poly)
            print(f"[INFO] Cargadas anotaciones previas para {img_path.name}")
        except Exception as e:
            print(f"[WARN] No se pudo cargar {j.name}: {e}")

def main():
    global img_orig, img_disp, polygons, current_points

    ap = argparse.ArgumentParser(description="Anotador manual de rocas con polígonos (instancias).")
    ap.add_argument("--images", type=str, required=True,
                    help="Carpeta con las imágenes a anotar (las 30 seleccionadas).")
    ap.add_argument("--out", type=str, required=False,
                    default=None,
                    help="Carpeta base de salida. Por defecto, crea 'masks' y 'annotations' junto a 'images'.")
    ap.add_argument("--start-index", type=int, default=0,
                    help="Índice de inicio (por si quieres continuar más tarde).")
    args = ap.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"[ERROR] No existe la carpeta: {images_dir}")
        return

    # salidas por defecto (junto a images)
    if args.out is None:
        out_base = images_dir.parent
    else:
        out_base = Path(args.out)
        out_base.mkdir(parents=True, exist_ok=True)

    out_masks = out_base / "masks"
    out_json  = out_base / "annotations"
    out_masks.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in EXTS])
    if not files:
        print(f"[AVISO] No hay imágenes en: {images_dir}")
        return

    i = max(0, min(args.start_index, len(files)-1))
    cv.namedWindow("Anotador de rocas", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Anotador de rocas", on_mouse)

    while 0 <= i < len(files):
        img_path = files[i]
        img = cv.imread(str(img_path))
        if img is None:
            print(f"[WARN] No se pudo leer {img_path.name}, salto.")
            i += 1
            continue

        # preparar estado
        img_orig = img.copy()
        polygons = []
        current_points = []

        # cargar si ya hay anotaciones
        load_previous_if_any(img_path, out_json)
        draw_overlay()

        while True:
            cv.imshow("Anotador de rocas", img_disp)
            k = cv.waitKey(10) & 0xFF

            if k == ord('u'):  # deshacer punto
                if current_points:
                    current_points.pop()
                    draw_overlay()

            elif k == ord('z'):  # deshacer polígono
                if polygons:
                    polygons.pop()
                    draw_overlay()

            elif k == ord('c'):  # limpiar polígono en edición
                current_points = []
                draw_overlay()

            elif k == ord('s'):  # guardar
                save_annotation(img_path, out_masks, out_json)

            elif k == ord('n'):  # guardar + siguiente
                save_annotation(img_path, out_masks, out_json)
                i += 1
                break

            elif k == ord('b'):  # guardar + volver
                save_annotation(img_path, out_masks, out_json)
                i = max(0, i - 1)
                break

            elif k == 27 or k == ord('q'):  # Esc o q -> guardar y salir
                save_annotation(img_path, out_masks, out_json)
                cv.destroyAllWindows()
                return

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
