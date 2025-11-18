import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd
import math

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def fit_ellipse_axes(contour):
    pts = contour.squeeze()
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.shape[0] >= 5:
        (_, _), (w, h), _ = cv.fitEllipse(contour)
        return float(max(w, h)), float(min(w, h))
    # fallback: rectángulo mínimo
    (_, _), (w, h), _ = cv.minAreaRect(contour)
    return float(max(w, h)), float(min(w, h))

def parse_label_line(tokens):
    """
    Acepta:
    - class x1 y1 x2 y2 ... xN yN                 (polígono)
    - class conf x1 y1 x2 y2 ...                   (polígono con confianza)
    - class xc yc w h                              (caja)
    - class conf xc yc w h                         (caja con confianza)
    Devuelve ('poly', [(x,y),...]) o ('box', (xc,yc,w,h)) con coords NORMALIZADAS [0,1].
    """
    if not tokens:
        return None, None
    # clase
    try:
        cls_id = int(float(tokens[0]))
    except:
        return None, None

    # heurística: ¿hay confianza?
    idx = 1
    if len(tokens) >= 3:
        try:
            # si tokens[1] es float en [0,1] pero el resto son pares, puede ser conf
            _conf = float(tokens[1])
            # si tras conf hay 4 números -> caja; si hay >=6 y par -> polígono
            rem = tokens[2:]
            if len(rem) == 4:  # caja
                vals = list(map(float, rem))
                return "box", tuple(vals)
            elif len(rem) >= 6 and len(rem) % 2 == 0:  # polígono
                vals = list(map(float, rem))
                pts = [(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]
                return "poly", pts
        except:
            pass

    # sin confianza
    rem = tokens[1:]
    if len(rem) == 4:  # caja
        try:
            vals = list(map(float, rem))
            return "box", tuple(vals)
        except:
            return None, None
    elif len(rem) >= 6 and len(rem) % 2 == 0:  # polígono
        try:
            vals = list(map(float, rem))
            pts = [(vals[i], vals[i+1]) for i in range(0, len(vals), 2)]
            return "poly", pts
        except:
            return None, None

    return None, None

def load_contours_for_image(labels_dir: Path, stem: str, w: int, h: int):
    """
    Construye contornos en píxeles a partir del .txt de YOLO.
    Busca primero labels/<stem>.txt, luego labels/images/<stem>.txt.
    """
    path1 = labels_dir / f"{stem}.txt"
    path2 = labels_dir / "images" / f"{stem}.txt"
    lab = path1 if path1.exists() else (path2 if path2.exists() else None)
    if lab is None:
        return []

    contours = []
    with open(lab, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            kind, data = parse_label_line(tokens)
            if kind == "poly":
                # data: lista de (x,y) normalizados
                pts = np.array([[x*w, y*h] for (x, y) in data], dtype=np.float32)
                pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                contours.append(pts)
            elif kind == "box":
                # data: (xc, yc, w, h) normalizados → polígono rectangular
                xc, yc, bw, bh = data
                cx, cy = xc * w, yc * h
                ww, hh = bw * w, bh * h
                x1, y1 = cx - ww / 2, cy - hh / 2
                x2, y2 = cx + ww / 2, cy + hh / 2
                rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                rect = np.round(rect).astype(np.int32).reshape(-1, 1, 2)
                contours.append(rect)
            else:
                continue
    return contours

def find_images(pred_dir: Path):
    """
    Retorna lista de imágenes en:
    - pred_dir/ (raíz), si existen
    - si no, en pred_dir/images/
    """
    imgs = sorted([p for p in pred_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if imgs:
        return imgs
    images_sub = pred_dir / "images"
    imgs2 = sorted([p for p in images_sub.iterdir() if p.suffix.lower() in IMG_EXTS]) if images_sub.exists() else []
    return imgs2

def main():
    ap = argparse.ArgumentParser(description="Postproceso YOLO-seg/box: medir L/A y clasificar laja/normal.")
    ap.add_argument("--pred-dir", type=str, required=True, help="Carpeta de predict de YOLO (p.ej. runs/segment/predict or predict2).")
    ap.add_argument("--out-csv", type=str, default=None, help="Ruta CSV de salida (default: <pred-dir>/mediciones_LA.csv).")
    ap.add_argument("--ratio-laja", type=float, default=1.5, help="Umbral L/A para laja (L/A > umbral).")
    ap.add_argument("--min-area", type=float, default=20.0, help="Área mínima (px) para considerar una instancia.")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    labels_dir = pred_dir / "labels"
    if not labels_dir.exists():
        print(f"[ERROR] No existe {labels_dir}. Re-ejecuta predict con save_txt=True.")
        return

    images = find_images(pred_dir)
    if not images:
        print(f"[AVISO] No se encontraron imágenes en {pred_dir} ni en {pred_dir/'images'}")
        return

    out_csv = Path(args.out_csv) if args.out_csv else (pred_dir / "mediciones_LA.csv")
    rows = []

    for im in images:
        img = cv.imread(str(im))
        if img is None:
            continue
        h, w = img.shape[:2]

        contours = load_contours_for_image(labels_dir, im.stem, w, h)
        if not contours:
            # imagen sin detecciones
            rows.append({"imagen": im.name, "id": 0, "area_px": 0, "long_px": 0, "ancho_px": 0,
                         "ratio_LA": 0, "clase": "sin_det"})
            continue

        for idx, c in enumerate(contours, start=1):
            area = float(cv.contourArea(c))
            if area < args.min_area:
                continue
            long_px, ancho_px = fit_ellipse_axes(c)
            ratio = long_px / max(ancho_px, 1e-6)
            clase = "laja" if ratio > args.ratio_laja else "normal"
            rows.append({
                "imagen": im.name,
                "id": idx,
                "area_px": area,
                "long_px": float(long_px),
                "ancho_px": float(ancho_px),
                "ratio_LA": float(ratio),
                "clase": clase
            })

    if not rows:
        print("[AVISO] No se generaron mediciones. ¿Hay detecciones y labels no vacíos?")
        # De todas formas creamos CSV vacío con encabezados para depurar
        pd.DataFrame(columns=["imagen","id","area_px","long_px","ancho_px","ratio_LA","clase"]).to_csv(out_csv, index=False)
        print(f"[OK] CSV vacío creado en: {out_csv}")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)  # detalle por instancia (deja tal cual)

    # ----- RESUMEN % laja/normal por imagen -----
    resumen = (
        df[df["id"] > 0]
        .groupby("imagen")["clase"]
        .value_counts(normalize=True)
        .rename("porcentaje")
        .mul(100)
        .reset_index()
    )
    resumen_pivot = (
        resumen.pivot(index="imagen", columns="clase", values="porcentaje")
        .fillna(0.0)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    # Formateo fijo "00,000" (coma decimal). Convertimos a string antes de exportar.
    num_cols = [c for c in resumen_pivot.columns if c != "imagen"]
    for c in num_cols:
        resumen_pivot[c] = resumen_pivot[c].map(lambda x: f"{x:0.3f}".replace(".", ","))

    resumen_csv = out_csv.with_name(out_csv.stem + "_resumen.csv")
    # Usamos separador ';' para que Excel en español lo abra directo con coma decimal
    resumen_pivot.to_csv(resumen_csv, index=False, sep=";")

    print(f"[OK] Detalle → {out_csv}")
    print(f"[OK] Resumen → {resumen_csv}")


if __name__ == "__main__":
    main()
