import io
import numpy as np
import pandas as pd
import cv2 as cv
import streamlit as st
from ultralytics import YOLO

from Src.preprocesar_imagenes import (
    warp_perspective,
    gray_world_white_balance,
    illumination_correction,
    clahe_on_lab,
    gamma_correction,
    unsharp_mask,
)

# -------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Clasificación de Rocas CN",
    layout="wide",
)

RATIO_LAJA_UMBRAL = 1.8  # <<< umbral pedido

# -------------------------------------------------------------------
# LOGIN SENCILLO (2 USUARIOS)
# -------------------------------------------------------------------
USERS = {
    "Pablo": "lalo",   # cambia estas claves
    "Victor": "mora",
}


def login():
    st.sidebar.header("Acceso de usuario")

    if "auth" not in st.session_state:
        st.session_state.auth = False
        st.session_state.user = None

    # Si ya está logueado, mostrar info + botón logout
    if st.session_state.auth:
        st.sidebar.success(f"Conectado como: {st.session_state.user}")
        if st.sidebar.button("Cerrar sesión"):
            st.session_state.auth = False
            st.session_state.user = None
            st.rerun()
        return True

    # Formulario de login
    username = st.sidebar.text_input("Usuario")
    password = st.sidebar.text_input("Clave", type="password")

    if st.sidebar.button("Ingresar"):
        if username in USERS and USERS[username] == password:
            st.session_state.auth = True
            st.session_state.user = username
            st.sidebar.success("Acceso concedido")
            st.rerun()
        else:
            st.sidebar.error("Usuario o clave incorrectos")

    return False


# Si no está autenticado detenemos la app
if not login():
    st.stop()

# -------------------------------------------------------------------
# CARGA MODELO YOLO
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # ajusta ruta si es necesario
    return model


model = load_model()

# -------------------------------------------------------------------
# PARÁMETROS PREPROCESADO / MEDICIÓN
# -------------------------------------------------------------------
st.title("Aplicación Streamlit - Clasificación de Rocas (CN)")

st.markdown(
    """
Sube una imagen nueva del harnero/correa.  
La app:

1. Aplica el **preprocesado** (warp + corrección de iluminación, etc.).
2. Ejecuta la **CNN / YOLO** (`best.pt`).
3. Muestra detecciones sobre la imagen preprocesada.
4. Genera una **tabla/Excel** con estimación de tamaño y tipo de roca (`laja`/`normal`).
"""
)

st.sidebar.subheader("Parámetros de preprocesado")

default_corners = "7,9;692,9;692,475;12,471"
warp_corners = st.sidebar.text_input(
    "Esquinas (warp_corners)",
    value=default_corners,
    help='Formato: "x1,y1;x2,y2;x3,y3;x4,y4"',
)

use_wb = st.sidebar.checkbox("White balance (Gray-World)", value=True)
bg_kernel = st.sidebar.slider("Kernel corrección iluminación (bg_kernel)", 3, 101, 61, step=2)
clahe_clip = st.sidebar.slider("CLAHE clipLimit", 1.0, 5.0, 3.0, step=0.1)
clahe_tiles = st.sidebar.slider("CLAHE tiles", 4, 16, 8, step=1)
gamma_val = st.sidebar.slider("Gamma", 0.5, 2.0, 1.1, step=0.05)
sharp_ksize = st.sidebar.slider("Kernel sharpen", 3, 11, 5, step=2)
sharp_amount = st.sidebar.slider("Intensidad sharpen", 0.0, 3.0, 1.0, step=0.1)

st.sidebar.subheader("Parámetros modelo YOLO")
conf_thres = st.sidebar.slider("Confianza mínima YOLO", 0.1, 0.9, 0.25, step=0.05)
iou_thres = st.sidebar.slider("IoU NMS", 0.1, 0.9, 0.45, step=0.05)
imgsz = st.sidebar.slider("Tamaño de entrada (imgsz)", 320, 1280, 640, step=64)

st.sidebar.subheader("Mediciones de tamaño")
mm_por_pixel = st.sidebar.number_input(
    "mm por pixel (escala)", min_value=0.01, max_value=10.0, value=2.0, step=0.01
)
min_area_px = st.sidebar.number_input(
    "Área mínima roca (px²)", min_value=0.0, value=5000.0, step=100.0
)

st.sidebar.info(f"Umbral laja (ratio largo/corto) = **{RATIO_LAJA_UMBRAL}**")


# -------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------------------------
def preprocess_frame(bgr: np.ndarray) -> np.ndarray:
    """Replica el pipeline de preprocesar_imagenes.py pero para un solo frame."""
    # 1) Warp
    if warp_corners and ";" in warp_corners:
        try:
            bgr_proc = warp_perspective(bgr, warp_corners, dst_w=None, dst_h=None)
        except Exception as e:
            st.warning(f"No se pudo aplicar warp_perspective: {e}. Se usa imagen sin warp.")
            bgr_proc = bgr.copy()
    else:
        bgr_proc = bgr.copy()

    # 2) White balance
    if use_wb:
        bgr_proc = gray_world_white_balance(bgr_proc)

    # 3) Iluminación (solo para obtener background, si quieres usarlo)
    gray = cv.cvtColor(bgr_proc, cv.COLOR_BGR2GRAY)
    _ = illumination_correction(gray, ksize=bg_kernel)

    # 4) CLAHE
    bgr_clahe = clahe_on_lab(bgr_proc, clip=clahe_clip, tiles=clahe_tiles)

    # 5) Gamma
    bgr_gamma = gamma_correction(bgr_clahe, gamma=gamma_val)

    # 6) Sharpen
    bgr_out = unsharp_mask(bgr_gamma, ksize=sharp_ksize, amount=sharp_amount)

    return bgr_out


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


def construir_tabla_mediciones(res, escala_mm_px: float, area_min_px: float) -> pd.DataFrame:
    """Construye DataFrame con métricas por roca y clasificación laja/normal."""
    boxes = res.boxes

    if boxes is None or boxes.shape[0] == 0:
        return pd.DataFrame()

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    registros = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        w = float(x2 - x1)
        h = float(y2 - y1)
        area = w * h
        if area < area_min_px:
            continue  # filtramos finos

        x_c = float((x1 + x2) / 2.0)
        y_c = float((y1 + y2) / 2.0)

        lado_largo = max(w, h)
        lado_corto = max(min(w, h), 1e-3)
        ratio = lado_largo / lado_corto

        # Clasificación geométrica laja/normal
        es_laja = ratio >= RATIO_LAJA_UMBRAL
        clase_LA = "laja" if es_laja else "normal"

        # Score asociado (usamos confianza YOLO en el string, como pediste)
        conf_det = float(confs[i])
        clase_LA_conf = f"{clase_LA} {conf_det:.2f}"

        # Escalamos a mm
        w_mm = w * escala_mm_px
        h_mm = h * escala_mm_px
        area_mm2 = area * (escala_mm_px**2)
        # Diámetro equivalente (círculo misma área)
        diam_eq_mm = 2.0 * np.sqrt(area_mm2 / np.pi)

        registros.append(
            {
                "id_rock": i,
                "x_c_px": x_c,
                "y_c_px": y_c,
                "ancho_px": w,
                "alto_px": h,
                "area_px": area,
                "ratio_largo_corto": ratio,
                "clase_LA": clase_LA,
                "clase_LA_conf": clase_LA_conf,  # ej. 'laja 0.81'
                "conf_yolo": conf_det,
                "ancho_mm": w_mm,
                "alto_mm": h_mm,
                "area_mm2": area_mm2,
                "diam_eq_mm": diam_eq_mm,
            }
        )

    df = pd.DataFrame(registros)
    return df


def generar_excel_mediciones(df_detalle: pd.DataFrame) -> bytes:
    """Genera archivo Excel en memoria con detalle + resumen laja/normal."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        # Detalle completo
        df_detalle.to_excel(writer, index=False, sheet_name="detalle_rocas")

        # Resumen laja vs normal (solo rocas sobre área mínima)
        resumen = (
            df_detalle["clase_LA"]
            .value_counts(normalize=True)
            .rename("porcentaje_%")
            .mul(100)
            .round(4)
            .reset_index()
            .rename(columns={"index": "clase_LA"})
        )
        resumen.to_excel(writer, index=False, sheet_name="resumen_laja_normal")

    buffer.seek(0)
    return buffer.getvalue()


# -------------------------------------------------------------------
# SUBIDA DE IMAGEN Y PIPELINE
# -------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Sube una imagen de rocas (jpg, png, ...)",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    if img_bgr is None:
        st.error("No se pudo leer la imagen. Prueba con otro archivo.")
    else:
        # ------------------ PREPROCESADO ------------------
        img_pre_bgr = preprocess_frame(img_bgr)

        # ------------------ YOLO ------------------
        with st.spinner("Ejecutando modelo..."):
            results = model(
                img_pre_bgr,
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False,
            )

        res = results[0]
        pred_bgr = res.plot()

        # ------------------ LAYOUT DE IMÁGENES ------------------
        col1, col2, col3 = st.columns(3)

        col1.subheader("1) Imagen original")
        col1.image(bgr_to_rgb(img_bgr), use_container_width=True)

        col2.subheader("2) Imagen preprocesada")
        col2.image(bgr_to_rgb(img_pre_bgr), use_container_width=True)

        col3.subheader("3) Detecciones modelo (CN)")
        col3.image(bgr_to_rgb(pred_bgr), use_container_width=True)

        # ------------------ TABLAS DE RESULTADOS ------------------
        st.markdown("---")
        st.subheader("Mediciones por roca (sobre área mínima)")

        df_med = construir_tabla_mediciones(
            res,
            escala_mm_px=mm_por_pixel,
            area_min_px=min_area_px,
        )

        if df_med.empty:
            st.warning("No se encontraron rocas sobre el área mínima especificada.")
        else:
            st.dataframe(df_med, use_container_width=True)

            # Resumen laja vs normal
            st.markdown("### Resumen laja vs normal (solo rocas sobre área mínima)")
            resumen = (
                df_med["clase_LA"]
                .value_counts(normalize=True)
                .rename("porcentaje_%")
                .mul(100)
                .round(4)
                .reset_index()
                .rename(columns={"index": "clase_LA"})
            )
            st.dataframe(resumen, use_container_width=True)

            # ------------------ DESCARGA EXCEL ------------------
            try:
                excel_bytes = generar_excel_mediciones(df_med)
                st.download_button(
                    label="📥 Descargar resultados en Excel",
                    data=excel_bytes,
                    file_name="mediciones_rocas_CN.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except ModuleNotFoundError as e:
                st.warning(
                    f"No se pudo generar la tabla de mediciones: {e}. "
                    "Instala el paquete 'xlsxwriter' en tu entorno."
                )
else:
    st.info("👆 Sube una imagen para comenzar.")



