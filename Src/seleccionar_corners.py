import argparse, cv2 as cv, numpy as np, pyperclip
from pathlib import Path

pts, img_show = [], None

def mouse_cb(event, x, y, flags, param):
    global pts, img_show
    if event == cv.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        cv.circle(img_show, (x, y), 5, (0,255,0), -1)
        cv.putText(img_show, str(len(pts)), (x+6, y-6), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.imshow("Selecciona 4 esquinas - Enter para confirmar", img_show)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="Ruta a una imagen de ejemplo")
    args = ap.parse_args()

    global img_show
    img = cv.imread(args.image)
    if img is None:
        print("[ERROR] No se pudo leer la imagen.")
        return

    img_show = img.copy()
    cv.namedWindow("Selecciona 4 esquinas - Enter para confirmar", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Selecciona 4 esquinas - Enter para confirmar", mouse_cb)
    cv.imshow("Selecciona 4 esquinas - Enter para confirmar", img_show)

    print("Haz clic en las 4 esquinas en orden (sup-izq, sup-der, inf-der, inf-izq).")
    print("Presiona Enter para terminar.")

    while True:
        key = cv.waitKey(1) & 255
        if key == 13:  # Enter
            break
        if key == 27:  # ESC
            cv.destroyAllWindows()
            return

    cv.destroyAllWindows()
    if len(pts) != 4:
        print(f"[ERROR] Se requieren 4 puntos; recibidos: {len(pts)}")
        return

    s = ";".join([f"{x},{y}" for (x,y) in pts])
    pyperclip.copy(s)
    print("\n--warp-corners")
    print(s)
    print("(Copiado al portapapeles ✅)")

    out_path = Path(args.image).with_name("corners_preview.png")
    show = img.copy()
    for i,(x,y) in enumerate(pts, start=1):
        cv.circle(show, (x,y), 6, (0,255,0), -1)
        cv.putText(show, f"{i}", (x+8,y-8), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv.imwrite(str(out_path), show)
    print(f"[OK] Guardado: {out_path}")

if __name__ == "__main__":
    main()
