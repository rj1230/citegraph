import requests, numpy as np
from src.data import load_cora

_, data = load_cora()
raw = (data.x.numpy() > 0).astype(float)
n = 500
mask = (data.edge_index[0] < n) & (data.edge_index[1] < n)
r = requests.post(
    "http://127.0.0.1:8000/predict/subgraph",
    json={
        "features": raw[:n].tolist(),
        "edge_index": data.edge_index[:, mask].tolist(),
    },
)
print(r.status_code)
preds = np.array([p["class_id"] for p in r.json()["predictions"]])
print("Subgraph accuracy vs labels:", (preds == data.y[:n].numpy()).mean())
