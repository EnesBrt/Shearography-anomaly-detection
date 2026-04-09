from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "detect_L2" / "train" / "weights" / "best.onnx"
SAMPLES_PATH = ROOT / "app_assets" / "sample_index.json"
LABELS = {
    "fault": "Fault",
    "good_clean": "Good Clean",
    "good_stripes": "Good Stripes",
    "uploaded": "Uploaded",
}

st.set_page_config(
    page_title="Shearography Defect Detection", page_icon="🔍", layout="wide"
)
st.markdown(
    """
    <style>
    :root {
        --app-bg: var(--background-color);
        --panel-bg: var(--secondary-background-color);
        --panel-border: rgba(127, 127, 127, 0.22);
        --panel-text: var(--text-color);
        --panel-subtle: color-mix(in srgb, var(--text-color) 72%, transparent);
        --metric-bg: var(--secondary-background-color);
        --metric-border: rgba(127, 127, 127, 0.22);
        --metric-text: var(--text-color);
        --shadow-strong: rgba(15,23,42,.10);
    }
    .block-container {max-width: 1380px; padding-top: 1.1rem; padding-bottom: 2rem;}
    .hero {
        background: linear-gradient(135deg, rgba(79,70,229,.14) 0%, rgba(99,102,241,.08) 100%);
        color: var(--panel-text); border-radius: 24px; padding: 1.25rem 1.35rem; margin-bottom: 1.35rem;
        box-shadow: 0 12px 30px var(--shadow-strong); border: 1px solid var(--panel-border);
    }
    .hero h1 {margin: .2rem 0 .45rem; font-size: 1.8rem; line-height: 1.1;}
    .hero p {margin: 0; max-width: 820px; color: var(--panel-subtle); line-height: 1.55;}
    .chip {
        display: inline-block; padding: .33rem .68rem; margin: 0 .42rem .45rem 0; border-radius: 999px;
        background: rgba(99,102,241,.12); color: var(--panel-text); font-size: .8rem; font-weight: 700;
        border: 1px solid rgba(99,102,241,.18);
    }
    .panel {
        background: var(--panel-bg); border: 1px solid var(--panel-border);
        border-radius: 20px; padding: 1.1rem; box-shadow: 0 12px 30px var(--shadow-strong);
        margin-bottom: 1.15rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: 20px;
        box-shadow: 0 12px 30px var(--shadow-strong);
    }
    .subtle {margin: 0; color: var(--panel-subtle); line-height: 1.45;}
    .section-title {font-size: 1.06rem; font-weight: 800; color: var(--panel-text);}
    .selection-note {
        margin: -.1rem 0 1rem 0;
        color: var(--panel-subtle);
        font-size: .96rem;
        line-height: 1.45;
    }
    .result-note {
        margin: -.15rem 0 .9rem 0;
        color: var(--panel-subtle);
        font-size: .94rem;
        line-height: 1.45;
    }
    .soft-spacer {height: .35rem;}
    .status {
        border-radius: 18px; padding: 1rem 1.1rem; color: white; margin-bottom: 1rem;
        box-shadow: 0 12px 28px rgba(15,23,42,.09);
    }
    .status strong {display: block; font-size: 1.12rem; margin-bottom: .15rem;}
    .fault {background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);}
    .clean {background: linear-gradient(135deg, #166534 0%, #22c55e 100%);}
    .stButton>button, .stDownloadButton>button {
        width: 100%; border-radius: 14px; font-weight: 700; padding: .82rem 1rem;
        border: 1px solid rgba(99,102,241,.18);
        box-shadow: 0 8px 18px rgba(15,23,42,.08);
    }
    [data-testid="stMetric"] {
        background: var(--metric-bg); border: 1px solid var(--metric-border); color: var(--metric-text);
        border-radius: 16px; padding: .9rem; box-shadow: 0 10px 24px var(--shadow-strong);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: var(--metric-text) !important;}
    [data-testid="stFileUploader"] {border-radius: 16px;}
    [data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model() -> YOLO:
    return YOLO(str(MODEL_PATH), task="detect")


@st.cache_data(show_spinner=False)
def get_samples() -> list[dict]:
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_image_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def open_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def sample_label(sample: dict) -> str:
    return f"{LABELS.get(sample['category'], sample['category'])} • {sample['label']}"


def predict(image: Image.Image, conf: float, iou: float, size: int) -> dict:
    result = get_model().predict(
        source=np.array(image), conf=conf, iou=iou, imgsz=size, verbose=False
    )[0]
    plotted = result.plot(line_width=3)[:, :, ::-1]
    boxes = result.boxes
    count = int(len(boxes))
    scores = boxes.conf.detach().cpu().numpy().astype(float) if count else np.array([])
    names = result.names if hasattr(result, "names") else {}
    rows = []
    for idx, (xyxy, score, cls_id) in enumerate(
        zip(
            boxes.xyxy.detach().cpu().numpy() if count else [],
            boxes.conf.detach().cpu().numpy() if count else [],
            boxes.cls.detach().cpu().numpy().astype(int) if count else [],
        ),
        start=1,
    ):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        rows.append(
            {
                "#": idx,
                "class": names.get(cls_id, str(cls_id)),
                "confidence": round(float(score), 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "width": round(x2 - x1, 1),
                "height": round(y2 - y1, 1),
            }
        )
    buf = io.BytesIO()
    Image.fromarray(plotted.astype(np.uint8)).save(buf, format="PNG")
    return {
        "image": plotted,
        "download": buf.getvalue(),
        "table": pd.DataFrame(
            rows,
            columns=[
                "#",
                "class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "width",
                "height",
            ],
        ),
        "count": count,
        "top": float(scores.max()) if count else None,
        "avg": float(scores.mean()) if count else None,
    }


def main() -> None:
    if not MODEL_PATH.exists() or not SAMPLES_PATH.exists():
        st.error("Model ou sample index introuvable dans le projet.")
        st.stop()

    st.markdown(
        """
        <div class="hero">
            <h1>Shearography Defect detection</h1>
            <p>Détection de défauts sur images de shearographie</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    conf, iou, size = 0.25, 0.45, 640

    sample_list = get_samples()
    image, meta = None, None

    with st.container(border=True):
        st.markdown(
            "<div class='section-title'>Sélection de l'image</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='selection-note'>Choisissez un exemple intégré ou téléversez votre propre image pour lancer l’analyse.</div>",
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "Source de l'image",
            ["Images intégrées", "Téléverser une image"],
            horizontal=True,
        )
        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
        if mode == "Images intégrées":
            category = st.segmented_control(
                "Type d'image",
                options=["fault", "good_clean", "good_stripes"],
                format_func=lambda x: LABELS[x],
                default="fault",
            )
            filtered = [s for s in sample_list if s["category"] == category]
            st.caption("Fault : Défaut réel")
            st.caption("Good clean : Image saine")
            st.caption("Good stripes : Image saine avec motifs de déformation")
            st.markdown("<div class='soft-spacer'></div>", unsafe_allow_html=True)
            meta = st.selectbox("Image test", filtered, format_func=sample_label)
            image = open_image(read_image_bytes(str(ROOT / meta["path"])))
        else:
            upload = st.file_uploader(
                "Televersez une image PNG / JPG / JPEG", type=["png", "jpg", "jpeg"]
            )
            if upload:
                image = Image.open(upload).convert("RGB")
                meta = {"label": upload.name, "category": "uploaded"}
            else:
                st.info("Ajoutez une image pour lancer l'analyse.")

    st.markdown("<div class='soft-spacer'></div>", unsafe_allow_html=True)
    run_col, spacer = st.columns([1.1, 2.2])
    with run_col:
        run = st.button("Lancer la détection", type="primary", disabled=image is None)

    if run and image is not None:
        with st.spinner("Exécution du modèle en cours..."):
            st.session_state["result"] = predict(image, conf, iou, size)
            st.session_state["input_image"] = image
            st.session_state["meta"] = meta
            st.session_state["settings"] = {
                "confidence_threshold": conf,
                "iou_threshold": iou,
                "image_size": size,
            }

    input_img = st.session_state.get("input_image", image)
    result = st.session_state.get("result")
    meta = st.session_state.get("meta", meta)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    preview_left, preview_right = st.columns(2, gap="large")
    with preview_left:
        with st.container(border=True):
            st.markdown(
                "<div class='section-title'>Image d'entrée</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='result-note'>Image source utilisée pour l'inférence.</div>",
                unsafe_allow_html=True,
            )
            if input_img is not None:
                st.image(
                    input_img,
                    use_container_width=True,
                )
            else:
                st.info("Sélectionne une image pour voir l'aperçu ici.")
    with preview_right:
        with st.container(border=True):
            st.markdown(
                "<div class='section-title'>Résultats</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='result-note'>L'image annotée apparaît ici après exécution du modèle.</div>",
                unsafe_allow_html=True,
            )
            if result:
                st.image(
                    result["image"],
                    use_container_width=True,
                )
            else:
                st.info("L'image annotée apparaîtra ici après exécution du modèle.")

    if not result:
        return

    status = (
        "<div class='status clean'><strong>✅ No fault detected</strong>Aucun défaut n'a été détecté avec les seuils actuels.</div>"
        if result["count"] == 0
        else f"<div class='status fault'><strong>⚠️ Fault detected</strong>{result['count']} défaut(s) détecté(s)</div>"
    )
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(status, unsafe_allow_html=True)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Défauts détectés", result["count"])
    metric_cols[1].metric(
        "Taux de confiance",
        f"{result['top']:.2%}" if result["top"] is not None else "—",
    )

    st.markdown("<div class='soft-spacer'></div>", unsafe_allow_html=True)
    dl_col, note_col = st.columns([1.15, 1.85])
    with dl_col:
        filename = f"{Path(meta['label'] if meta else 'prediction').stem}_annotated.png"
        st.download_button(
            "Télécharger l'image annotée",
            result["download"],
            file_name=filename,
            mime="image/png",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
