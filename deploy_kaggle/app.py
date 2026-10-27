from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.kaggle_heartbeat import KaggleHeartbeatPredictor

st.set_page_config(page_title="ECG AI Demo", page_icon="❤️")
st.title("ECG AI — Heartbeat Classification")
st.warning("Research and education only. Not for clinical diagnosis.")

st.write(
    "Upload a MIT-BIH Kaggle CSV containing heartbeat rows, "
    "or a CSV/TXT containing one heartbeat with 187 signal values."
)

@st.cache_resource
def get_model():
    return KaggleHeartbeatPredictor()

uploaded = st.file_uploader(
    "Upload heartbeat CSV or TXT",
    type=["csv", "txt"]
)

if uploaded:
    try:
        # Read only a small preview initially to avoid loading a huge dataset.
        preview = pd.read_csv(uploaded, header=None, nrows=100)
        st.write(f"Preview of the first {len(preview)} rows")

        row_index = st.number_input(
            "Heartbeat row to classify (starting at 1)",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )

        row = preview.iloc[int(row_index) - 1]
        values = pd.to_numeric(row, errors="coerce").to_numpy()

        # Kaggle MIT-BIH rows contain 187 signal values and a final label.
        if len(values) == 188:
            values = values[:187]

        if len(values) != 187 or not np.isfinite(values).all():
            st.error("This row does not contain 187 valid signal values.")
        else:
            result = get_model().predict(values.astype(np.float32))

            st.subheader("Prediction")
            st.metric("Predicted class", result["predicted_class"])
            st.metric("Model confidence", f'{result["confidence"]:.1%}')

            st.subheader("Class probabilities")
            st.bar_chart(pd.Series(result["probabilities"], name="Probability"))

            st.caption(
                "Classes: N=normal, S=supraventricular ectopic, "
                "V=ventricular ectopic, F=fusion, Q=paced/unknown category. "
                "Model confidence is not clinical certainty."
            )
    except Exception as exc:
        st.error(f"Could not process the file: {exc}")
