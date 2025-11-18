import cv2 as cv
import numpy as np
import pandas as pd

def draw_annotations(bgr, contours, df, scores=None):
    out = bgr.copy()
    for i, c in enumerate(contours):
        mask = np.zeros(bgr.shape[:2], dtype="uint8")
        cv.drawContours(mask, [c], -1, 255, -1)
        color = (255, 0, 0) if df.iloc[i]["clase"] == "laja" else (0, 200, 0)
        out[mask==255] = cv.addWeighted(out, 0.5, (np.dstack([mask]*3)>0).astype("uint8")*np.array(color, dtype="uint8"), 0.5, 0)[mask==255]
        # elipse
        ellipse = cv.fitEllipse(c) if len(c)>=5 else None
        if ellipse: cv.ellipse(out, ellipse, color, 2)
        # etiqueta
        x,y,w,h = cv.boundingRect(c)
        txt = f'{df.iloc[i]["clase"]} LA={df.iloc[i]["ratio_LA"]:.2f}'
        if scores is not None:
            txt = f'rock {scores[i]:.2f} | {txt}'
        cv.rectangle(out, (x,y), (x+w, y+24), color, -1)
        cv.putText(out, txt, (x+4, y+17), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv.LINE_AA)
    return out

def summarize_df(df):
    if df.empty:
        return pd.DataFrame([{"total":0,"laja_pct":0.0,"normal_pct":0.0}])
    total = len(df)
    laja = int((df["clase"]=="laja").sum())
    normal = total - laja
    laja_pct = 100.0*laja/total
    normal_pct = 100.0*normal/total
    return pd.DataFrame([{"total":total,"laja_pct":laja_pct,"normal_pct":normal_pct}])
