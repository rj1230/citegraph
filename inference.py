"""ONNX inference that mirrors api/app.py, so no FastAPI server is needed."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "artifacts" / "simple_gcn_cora.onnx"


@st.cache_resource
def get_session():
    return ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])


def normalize(x: np.ndarray) -> np.ndarray:
    # same as PyG T.NormalizeFeatures(): divide rows by their sum (min 1)
    return x / np.clip(x.sum(axis=1, keepdims=True), 1, None)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def predict(raw: np.ndarray, edge_index: np.ndarray) -> list:
    """raw: [n, 1433] 0/1 features. edge_index: [2, e], both directions."""
    logits = get_session().run(
        None,
        {
            "node_features": normalize(raw.astype(np.float32)).astype(np.float32),
            "edge_indices": edge_index.astype(np.int64),
        },
    )[0]
    p = softmax(logits)
    ids = p.argmax(axis=1)
    return [
        {"node": i, "class_id": int(k), "confidence": round(float(p[i, k]), 4)}
        for i, k in enumerate(ids)
    ]
