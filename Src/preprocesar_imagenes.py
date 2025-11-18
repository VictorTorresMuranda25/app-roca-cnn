import argparse
from pathlib import Path
import cv2 as cv
import numpy as np

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_corners(corners_str: str):
    pts = []
    for pair in corners_str.split(";"):
        x, y = pair.split(",")
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)

def estimate_dst_size(src_pts):
    def dist(a,b): return np.linalg.norm(a-b)
    w = int(max(dist(src_pts[0], src_pts[1]), dist(src_pts[2], src_pts[3])))
    h = int(max(dist(src_pts[1], src_pts[2]), dist(src_pts[3], src_pts[0])))
    return w, h

def warp_perspective(bgr, corners_str, dst_w=None, dst_h=None):
    src_pts = parse_corners(corners_str)
    if dst_w is None or dst_h is None:
        dst_w, dst_h = estimate_dst_size(src_pts)
    dst_pts = np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype=np.float32)
    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    return cv.warpPerspective(bgr, H, (dst_w, dst_h))

def gray_world_white_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    mean = img.reshape(-1,3).mean(axis=0) + 1e-6
    scale = mean.mean() / mean
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

def illumination_correction(gray, ksize=61):
    if ksize <= 1: return gray
    k = ksize if ksize % 2 == 1 else ksize + 1
    bg = cv.medianBlur(gray, k)
    norm = cv.subtract(gray, bg)
    return cv.normalize(norm, None, 0, 255, cv.NORM_MINMAX)

def clahe_on_lab(bgr, clip=3.0, tiles=8):
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    L, A, B = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    L2 = clahe.apply(L)
    return cv.cvtColor(cv.merge([L2, A, B]), cv.COLOR_LAB2BGR)

def gamma_correction(img, gamma=1.0):
    if gamma <= 0: gamma = 1.0
    inv = 1.0 / gamma
    tbl = np.array([(i/255.0)**inv * 255 for i in range(256)], dtype=np.uint8)
    return cv.LUT(img, tbl)

def unsharp_mask(bgr, ksize=5, amount=1.2):
    k = ksize if ksize % 2 == 1 else ksize + 1
    blur = cv.GaussianBlur(bgr, (k, k), 0)
    return cv.addWeighted(bgr, 1+amount, blur, -amount, 0)


def process_one(img_path: Path, out_dir: Path, dbg_dir: Path, args):
    bgr = cv.imread(str(img_path))
    if bgr is None:
        print(f"[WARN] No se pudo leer {img_path.name}")
        return

    orig = bgr.copy()

    if args.warp_corners:
        bgr = warp_perspective(bgr, args.warp_corners, args.dst_width, args.dst_height)
    if args.wb:
        bgr = gray_world_white_balance(bgr)

    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray_corr = illumination_correction(gray, ksize=args.bg_kernel)
    bgr_clahe = clahe_on_lab(bgr, clip=args.clahe_clip, tiles=args.clahe_tiles)
    bgr_gamma = gamma_correction(bgr_clahe, gamma=args.gamma)
    bgr_out = unsharp_mask(bgr_gamma, ksize=args.sharp_ksize, amount=args.sharp_amount)

    out_path = out_dir / img_path.name
    cv.imwrite(str(out_path), bgr_out)

    if args.debug:
        # Izquierda = ORIGINAL rectificada si hubo warp (para comparar manzana con manzana)
        left = warp_perspective(orig, args.warp_corners, args.dst_width, args.dst_height) if args.warp_corners else orig
        right = bgr_out

        # Redimensiona ambos a la misma ALTURA manteniendo aspecto
        def resize_to_h(img, target_h):
            h0, w0 = img.shape[:2]
            if h0 == target_h:
                return img
            new_w = int(round(w0 * (target_h / float(h0))))
            return cv.resize(img, (new_w, target_h), interpolation=cv.INTER_AREA)

        hL, wL = left.shape[:2]
        hR, wR = right.shape[:2]
        target_h = max(hL, hR)

        left_r = resize_to_h(left, target_h)
        right_r = resize_to_h(right, target_h)

        comp = np.zeros((target_h, left_r.shape[1] + right_r.shape[1], 3), dtype=np.uint8)
        comp[:, :left_r.shape[1]] = left_r
        comp[:, left_r.shape[1]:left_r.shape[1] + right_r.shape[1]] = right_r

        cv.putText(comp, "IZQ: Rectificada", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(comp, "DER: Mejorada", (left_r.shape[1] + 20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv.imwrite(str(dbg_dir / f"{img_path.stem}_compare.png"), comp)
        cv.imwrite(str(dbg_dir / f"{img_path.stem}_graycorr.png"), gray_corr)


def main():
    ap = argparse.ArgumentParser(description="Preprocesado por lotes con rectificación (homografía opcional).")
    ap.add_argument("--input",  type=str, default=r"C:\Users\victo\OneDrive\Documentos\Preprocesador imagenes rocas\imagenes_originales")
    ap.add_argument("--output", type=str, default=r"C:\Users\victo\OneDrive\Documentos\Preprocesador imagenes rocas\imagenes_mejoradas")
    ap.add_argument("--warp-corners", type=str, default=None, help='Ejemplo: "123,45;987,44;990,520;120,530"')
    ap.add_argument("--dst-width",  type=int, default=None)
    ap.add_argument("--dst-height", type=int, default=None)
    ap.add_argument("--wb", action="store_true")
    ap.add_argument("--bg-kernel", type=int, default=61)
    ap.add_argument("--clahe-clip", type=float, default=3.0)
    ap.add_argument("--clahe-tiles", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=1.1)
    ap.add_argument("--sharp-ksize", type=int, default=5)
    ap.add_argument("--sharp-amount", type=float, default=1.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    dbg_dir = out_dir / "_debug" if args.debug else None
    ensure_dir(out_dir)
    if args.debug: ensure_dir(dbg_dir)

    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if not files:
        print(f"[AVISO] No hay imágenes en: {in_dir}")
        return

    for p in files:
        process_one(p, out_dir, dbg_dir, args)

    print(f"[OK] Imágenes mejoradas guardadas en: {out_dir}")
    if args.debug: print(f"[DEBUG] Comparativas guardadas en: {dbg_dir}")

if __name__ == "__main__":
    main()
