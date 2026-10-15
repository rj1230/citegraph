from contextlib import asynccontextmanager
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import load_cora

ONNX_PATH = "artifacts/simple_gcn_cora.onnx"
NUM_FEATURES = 1433
MAX_NODES, MAX_EDGES = 5000, 50000
CLASSES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]

state = {}


def normalize(x: np.ndarray) -> np.ndarray:
    # identical to PyG T.NormalizeFeatures(): divide rows by their sum (min 1)
    return x / np.clip(x.sum(axis=1, keepdims=True), 1, None)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def to_result(logits_row: np.ndarray) -> dict:
    p = softmax(logits_row)
    k = int(p.argmax())
    return {
        "class_id": k,
        "class_name": CLASSES[k],
        "confidence": round(float(p[k]), 4),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the session once, and precompute the full-graph logits for the lookup path
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    _, data = load_cora()
    logits = sess.run(
        None, {"node_features": data.x.numpy(), "edge_indices": data.edge_index.numpy()}
    )[0]
    state.update(sess=sess, logits=logits)
    yield
    state.clear()


app = FastAPI(title="Cora GCN Service", lifespan=lifespan)


class Subgraph(BaseModel):
    # raw binary word indicators, one row of 1433 values per paper
    features: List[List[float]] = Field(
        ..., description="[num_nodes][1433] raw 0/1 features"
    )
    # [[sources...], [targets...]]; send both directions for undirected citations
    edge_index: List[List[int]] = Field(..., description="[2][num_edges]")


@app.get("/health")
def health():
    return {"status": "ok", "nodes_cached": int(state["logits"].shape[0])}


@app.get("/predict/node/{node_id}")
def predict_node(node_id: int):
    logits = state["logits"]
    if not 0 <= node_id < logits.shape[0]:
        raise HTTPException(404, f"node_id must be in [0, {logits.shape[0] - 1}]")
    return {"node_id": node_id, **to_result(logits[node_id])}


@app.post("/predict/subgraph")
def predict_subgraph(body: Subgraph):
    x = np.asarray(body.features, dtype=np.float32)
    ei = np.asarray(body.edge_index, dtype=np.int64)

    if x.ndim != 2 or x.shape[1] != NUM_FEATURES:
        raise HTTPException(422, f"each feature row must have {NUM_FEATURES} values")
    n = x.shape[0]
    if not 1 <= n <= MAX_NODES:
        raise HTTPException(422, f"num_nodes must be in [1, {MAX_NODES}]")
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise HTTPException(422, "edge_index must have shape [2, num_edges]")
    if ei.shape[1] > MAX_EDGES:
        raise HTTPException(422, f"num_edges must be <= {MAX_EDGES}")
    if ei.size and (ei.min() < 0 or ei.max() >= n):
        raise HTTPException(422, "edge_index contains node ids outside [0, num_nodes)")
    if not np.isin(x, (0.0, 1.0)).all():
        raise HTTPException(
            422,
            "features must be raw 0/1 indicators (normalization is applied server-side)",
        )

    logits = state["sess"].run(
        None, {"node_features": normalize(x).astype(np.float32), "edge_indices": ei}
    )[0]
    return {"predictions": [{"node": i, **to_result(logits[i])} for i in range(n)]}
