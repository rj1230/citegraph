# ⚡ CiteGraph

## Production-Oriented Citation-Graph Classification

Classify research papers in the Cora citation network using a Graph Convolutional Network (GCN), then serve the trained model through ONNX Runtime, FastAPI, and an interactive Streamlit dashboard.

🔗 **GitHub:** [citegraph](https://github.com/rj1230/citegraph)

---

## 📌 Overview

CiteGraph is an end-to-end graph machine learning system for research-topic classification on the **Cora citation network**.

Each paper is represented as a node with 1,433 bag-of-words features, while citations form edges between papers. A two-layer GCN learns from both paper content and local citation structure to predict one of seven research topics.

The project is not just a notebook model. It includes reproducible multi-seed evaluation, model comparison, ONNX export with numerical validation, a FastAPI inference service, and a Streamlit interface for exploring how citation structure affects predictions.

---

## ✨ Key Capabilities

### 🧠 Graph-Aware Classification

- Two-layer GCN implemented with PyTorch Geometric.
- Uses both node features and citation edges for classification.
- Classifies 2,708 Cora papers into 7 research topics.
- Achieves **82.5% ± 0.8** test accuracy across 10 random seeds.

### 📊 Reproducible Benchmarking

- Benchmarks MLP, GCN, GAT, and APPNP over 10 seeds.
- Reports mean accuracy and standard deviation rather than single-run results.
- Compares graph-based models against a feature-only MLP baseline.
- GCN improves accuracy by **24.1 percentage points** over the MLP baseline.

### 🚀 ONNX Inference

- Exports the trained GCN to ONNX with dynamic node and edge dimensions.
- Validates ONNX Runtime outputs against PyTorch on the full graph and a 1,000-node subgraph.
- Maximum output difference between PyTorch and ONNX: approximately **1e-6**.

### 🌐 FastAPI Serving

- Serves the ONNX model through ONNX Runtime.
- Provides cached full-graph Cora predictions.
- Supports custom subgraph inference.
- Normalizes features server-side.
- Validates incoming requests before inference.

### 🖥️ Interactive Streamlit Dashboard

- Sends paper neighborhoods to the FastAPI service.
- Displays predicted topic, class probabilities, and confidence.
- Includes a citation-dropout slider to demonstrate the contribution of graph structure.
- Visualizes benchmark results with error bars.

---

## 📊 Validation Snapshot

| Result | Value |
|---|---:|
| Dataset | Cora |
| Nodes / papers | 2,708 |
| Citation edges | 10,556 |
| Node features | 1,433 |
| Classes | 7 |
| Train / validation / test nodes | 140 / 500 / 1,000 |
| Evaluation seeds | 10 |
| Feature-only MLP accuracy | 58.4% |
| **GCN accuracy** | **82.5% ± 0.8** |
| APPNP accuracy | 83.1% |
| GCN improvement over MLP | **24.1 percentage points** |
| ONNX vs. PyTorch max output difference | ~1e-6 |

APPNP achieved the highest mean accuracy, but its improvement over GCN was within approximately one standard deviation. GCN was selected for deployment because it offers strong accuracy, a simpler architecture, and more reliable ONNX export.

---

## 🏗️ Architecture

### Request Flow

```text
Cora Dataset
     |
     v
Feature Normalization
     |
     v
PyTorch Geometric GCN
     |
     v
ONNX Export
     |
     v
FastAPI + ONNX Runtime
     |
     v
Streamlit Dashboard
```

### Model Pipeline

1. Load the Cora citation graph through PyTorch Geometric.
2. Normalize node features.
3. Train a two-layer GCN using the standard Cora train/validation/test split.
4. Apply early stopping using best-validation weights.
5. Evaluate across 10 random seeds.
6. Export the best model to ONNX.
7. Validate ONNX outputs against PyTorch.
8. Serve predictions through FastAPI and ONNX Runtime.
9. Explore predictions through Streamlit.

---

## 🎯 Why Graph Structure Matters

A feature-only MLP sees only the words in each paper. It cannot use the fact that papers citing similar research are often part of the same topic community.

The GCN uses both:

- **Node features:** 1,433 binary word-presence features.
- **Graph structure:** citation links between papers.

This allows the model to propagate information across the citation network and produce substantially better topic predictions.

| Model | Uses Citation Graph | Accuracy |
|---|---|---:|
| MLP | No | 58.4% |
| **GCN** | **Yes** | **82.5% ± 0.8** |
| GAT | Yes | Benchmarked |
| APPNP | Yes | 83.1% |

---

## 🔧 Engineering Notes

### Checkpointing Fix

During training, early stopping initially restored the latest model weights instead of the best-validation weights.

The issue occurred because `state_dict()` returns live references to model tensors. After fixing checkpoint restoration and combining it with feature normalization and regularization, single-run accuracy improved from **78.4% to 82.6%**.

### Model Selection

APPNP achieved the best mean accuracy at **83.1%**, but the difference from GCN was within roughly one standard deviation.

GCN was selected for deployment because it:

- Achieved nearly equivalent accuracy.
- Has a simpler architecture.
- Exports more reliably to ONNX.
- Is easier to serve and maintain.

### ONNX Parity

The ONNX model was tested on:

- The full Cora graph.
- A 1,000-node subgraph.

The maximum difference between PyTorch and ONNX Runtime outputs was approximately `1e-6`, confirming that the exported model preserves numerical behavior.

---

## 📡 API

### Start the API

```bash
uvicorn api.app:app --reload
```

Open interactive documentation:

```text
http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API and model availability |
| GET | `/model-info` | Return model and dataset metadata |
| POST | `/predict` | Run inference on a provided graph and node indices |
| POST | `/predict/cora_node` | Run inference for selected Cora nodes |

---

## 🖥️ Streamlit Dashboard

Start the dashboard:

```bash
streamlit run frontend/ui.py
```

The dashboard provides:

- Paper-node selection and neighborhood construction.
- Real-time inference through the FastAPI service.
- Predicted topic and probability distribution.
- Citation-dropout controls.
- Benchmark comparison with error bars.
- A visual demonstration of graph contribution to classification.

---

## 🧩 Tech Stack

| Layer | Tool |
|---|---|
| Graph model | PyTorch Geometric GCN |
| Deep-learning framework | PyTorch |
| Model export | ONNX |
| Inference runtime | ONNX Runtime |
| API | FastAPI |
| Dashboard | Streamlit |
| Evaluation | scikit-learn, NumPy, Pandas |
| Dataset | Cora / Planetoid |

---

## 📂 Project Structure

```text
citegraph/
├── api/
│   └── app.py               # FastAPI ONNX inference service
├── artifacts/
│   ├── gcn.pt               # Trained PyTorch model
│   └── simple_gcn_cora.onnx # Exported ONNX model
├── frontend/
│   └── ui.py                # Streamlit dashboard
├── notebooks/
│   └── main.ipynb           # EDA, training, and evaluation
├── src/
│   ├── data.py              # Data loading and preprocessing
│   ├── models.py            # GCN architecture
│   ├── train.py             # Training and evaluation
│   └── export_onnx.py       # ONNX export
├── tests/
│   └── check_api.py         # API checks
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone [https://github.com/rj1230/citegraph.git](https://github.com/rj1230/citegraph.git)
cd citegraph
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

### 3. Export to ONNX

```bash
python src/export_onnx.py
```

### 4. Start the API

```bash
uvicorn api.app:app --reload
```

### 5. Start the dashboard

```bash
streamlit run frontend/ui.py
```

---

## 🧭 Design Principles

- **Graph-aware, not feature-only** — the system explicitly measures the value added by citation structure.
- **Reproducible evaluation** — results are reported across 10 seeds with mean and standard deviation.
- **Deployment-first** — the trained model is exported, validated, and served through a real inference API.
- **Explainable predictions** — the dashboard shows probabilities, confidence, and the effect of removing citation edges.
- **Portable inference** — ONNX decouples trained PyTorch modeling from the serving runtime.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 GitHub

[@rj1230](https://github.com/rj1230)
