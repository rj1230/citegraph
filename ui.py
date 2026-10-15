"""Cora GCN Explorer: single-file Streamlit UI (Streamlit Cloud ready).

The ONNX model runs inside this app, so no separate API server is needed.

Run locally:
    streamlit run ui.py
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

from inference import ONNX_PATH, get_session, predict

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "artifacts" / "results.csv"
DATA_NPZ = ROOT / "artifacts" / "cora.npz"

# Fill this in once you have verified the label order, e.g. {0: "Theory", ...}
CLASS_NAMES: dict = {}

st.set_page_config(page_title="Cora GCN Explorer", page_icon="🕸️", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
#MainMenu, footer {visibility: hidden;}

.hero {padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.1rem;
       background: linear-gradient(135deg, #14213d 0%, #1b2b4d 55%, #17364a 100%);
       border: 1px solid #2a3b5c;}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;
          background: linear-gradient(90deg, #7db2ff, #2dd4bf);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {margin: .45rem 0 0 0; color: #b6c4da; font-size: 1.02rem;}

.stat {background: #121a27; border: 1px solid #243047; border-radius: 12px;
       padding: .8rem 1rem;}
.stat .v {font-size: 1.5rem; font-weight: 700; color: #e6edf3; line-height: 1.2;}
.stat .l {font-size: .78rem; color: #8fa0b8; text-transform: uppercase;
          letter-spacing: .6px;}

.chip {display: inline-block; width: 12px; height: 12px; border-radius: 50%;
       margin-right: 8px; vertical-align: middle;}
.badge {display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: .78rem; font-weight: 600;}
.ok {background: #12382b; color: #4ade80;}
.bad {background: #411a1f; color: #f87171;}
.bar {height: 8px; border-radius: 999px; background: #223046;}
.bar > div {height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, #4f8bf9, #2dd4bf);}

.grid4 {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;}
.grid2 {display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;}
.kpi {background: #121a27; border: 1px solid #243047; border-radius: 14px;
      padding: 1rem 1.15rem; display: flex; flex-direction: column; min-height: 172px;}
.kpi.sm {min-height: 112px;}
.kpi .t {font-size: .78rem; color: #8fa0b8; text-transform: uppercase;
         letter-spacing: .7px; margin-bottom: .45rem;}
.kpi .big {font-size: 1.55rem; font-weight: 700; color: #e6edf3; line-height: 1.3;}
.kpi .foot {margin-top: auto; padding-top: .6rem;}
.kpi .sub {font-size: .8rem; color: #9fb0c8; margin-top: .4rem;}
.kpi .bar {margin-top: .6rem;}

.section {font-size: 1.05rem; font-weight: 700; color: #e6edf3;
          margin: 1.2rem 0 .5rem 0;}
.note {background: #131c2b; border-left: 3px solid #4f8bf9; border-radius: 6px;
       padding: .7rem 1rem; color: #b6c4da; font-size: .92rem;}
</style>
""",
    unsafe_allow_html=True,
)

PALETTE = [to_hex(plt.cm.tab10(i)) for i in range(10)]


def cls_label(c: int) -> str:
    return CLASS_NAMES.get(c, f"Class {c}")


# ----------------------------------------------------------------------------
# Data (pre-exported from PyG, so the cloud app needs no torch)
# ----------------------------------------------------------------------------
missing = [p.name for p in (ONNX_PATH, DATA_NPZ) if not p.exists()]
if missing:
    st.error(
        f"Missing file(s) in artifacts/: {', '.join(missing)}. "
        "Commit them to the repo (and check .gitignore). "
        "Create cora.npz locally with `python export_data.py`."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading Cora...")
def get_data():
    d = np.load(DATA_NPZ)
    return d["raw"], d["edge_index"], d["y"]


raw, edge_index, y = get_data()
N = raw.shape[0]
NUM_FEATURES = raw.shape[1]
NUM_EDGES = edge_index.shape[1]
NUM_CLASSES = int(y.max()) + 1


@st.cache_resource(show_spinner="Running the model on the full graph...")
def full_graph_preds():
    """Replaces the API's cached full-graph logits."""
    return predict(raw, edge_index)


def k_hop_subgraph(node: int, hops: int, ei: np.ndarray, n: int):
    """NumPy version of PyG's k_hop_subgraph (relabelled, sorted node ids)."""
    in_set = np.zeros(n, dtype=bool)
    in_set[node] = True
    frontier = in_set.copy()
    for _ in range(hops):
        new = np.zeros(n, dtype=bool)
        new[ei[0][frontier[ei[1]]]] = True
        frontier = new & ~in_set
        in_set |= new
    subset = np.flatnonzero(in_set)
    sub_ei = ei[:, in_set[ei[0]] & in_set[ei[1]]]
    remap = -np.ones(n, dtype=np.int64)
    remap[subset] = np.arange(len(subset))
    return subset, remap[sub_ei], int(remap[node])


def parse_pm(s: str):
    mean, std = s.split("±")
    return float(mean), float(std)


def show_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=200, bbox_inches="tight")
    st.image(buf)
    plt.close(fig)


def best_accuracy():
    if not RESULTS_CSV.exists():
        return None
    df_ = pd.read_csv(RESULTS_CSV)
    return max(parse_pm(s)[0] for s in df_["Accuracy"])


def stat_box(col, value, label):
    col.markdown(
        f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def kpi(title, big, sub="", badge="", pct=None, small=False):
    bar = (
        f'<div class="bar"><div style="width:{pct:.1f}%"></div></div>'
        if pct is not None
        else ""
    )
    cls = "kpi sm" if small else "kpi"
    return (
        f'<div class="{cls}"><div class="t">{title}</div><div class="big">{big}</div>'
        f'<div class="foot">{badge}{bar}<div class="sub">{sub}</div></div></div>'
    )


def class_big(c: int) -> str:
    return f'<span class="chip" style="background:{PALETTE[c]}"></span>{cls_label(c)}'


def pred_kpi(title, pred, true_id):
    ok = pred["class_id"] == true_id
    badge = (
        '<span class="badge ok">correct</span>'
        if ok
        else '<span class="badge bad">wrong</span>'
    )
    pct = min(max(pred["confidence"], 0.0), 1.0) * 100
    return kpi(title, class_big(pred["class_id"]), f"Confidence {pct:.1f}%", badge, pct)


# ----------------------------------------------------------------------------
# Sidebar: model status
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model status")
    try:
        get_session()
        st.success("Model loaded (ONNX Runtime)")
    except Exception as e:  # noqa: BLE001
        st.error("Model failed to load")
        st.code(str(e))
    st.caption(f"{N:,} nodes · {NUM_EDGES:,} directed edges")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Cora GCN Explorer</h1>
  <p>Classify research papers by topic from their text and their citation
  neighborhood. A two-layer GCN running on ONNX Runtime.</p>
</div>
""",
    unsafe_allow_html=True,
)

best_acc = best_accuracy()
s1, s2, s3, s4, s5 = st.columns(5)
stat_box(s1, f"{N:,}", "Papers")
stat_box(s2, f"{NUM_EDGES:,}", "Citation edges")
stat_box(s3, f"{NUM_FEATURES:,}", "Word features")
stat_box(s4, NUM_CLASSES, "Topics")
stat_box(s5, f"{best_acc:.1%}" if best_acc else "n/a", "Best test accuracy")
st.write("")

tab_explorer, tab_results, tab_about = st.tabs(
    ["🔎 Paper explorer", "📊 Model results", "📐 About"]
)


# ----------------------------------------------------------------------------
# Tab 1: Paper explorer
# ----------------------------------------------------------------------------
with tab_explorer:
    with st.container(border=True):
        k1, k2, k3, k4 = st.columns([2, 2, 3, 2])
        node_id = int(k1.number_input("Paper (node) ID", 0, N - 1, 0, step=1))
        hops = k2.radio("Neighborhood depth (hops)", [1, 2], horizontal=True)
        keep = k3.slider("Citations kept (%)", 0, 100, 100, 10)
        seed = int(k4.number_input("Random seed", 0, 9999, 0))

    # Build the ego subgraph and thin out its citations
    subset, ei, target = k_hop_subgraph(node_id, hops, edge_index, N)
    und = ei[:, ei[0] < ei[1]]  # one row per undirected edge
    rng = np.random.default_rng(seed)
    n_keep = int(round(und.shape[1] * keep / 100))
    und = und[:, rng.permutation(und.shape[1])[:n_keep]]
    ei_used = np.concatenate([und, und[::-1]], axis=1)  # mirror to both directions
    direct_citations = int((edge_index[0] == node_id).sum())

    full = full_graph_preds()[node_id]
    sub_pred = predict(raw[subset], ei_used)[target]
    true_id = int(y[node_id])

    st.markdown('<div class="section">Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="grid4">'
        + kpi("True label", class_big(true_id), "Ground truth from the dataset")
        + pred_kpi("Full-graph prediction", full, true_id)
        + pred_kpi("Ego-graph prediction", sub_pred, true_id)
        + kpi(
            "Direct citations",
            f"{direct_citations}",
            f"{len(subset)} papers in the {hops}-hop view",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if full["class_id"] != sub_pred["class_id"]:
        st.markdown(
            '<div class="note" style="margin-top:.8rem">The two predictions differ. '
            "With fewer neighbors the model has less evidence to work with.</div>",
            unsafe_allow_html=True,
        )

    # Neighborhood drawing
    G = nx.Graph()
    G.add_nodes_from(range(len(subset)))
    G.add_edges_from(und.T.tolist())
    sub_labels = y[subset]
    colors = [PALETTE[int(c)] for c in sub_labels]
    sizes = [520 if i == target else 55 for i in range(len(subset))]
    outlines = ["white" if i == target else "none" for i in range(len(subset))]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.22, edge_color="#8899aa")
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        edgecolors=outlines,
        linewidths=2,
    )
    present = sorted(set(int(c) for c in sub_labels))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=PALETTE[c],
            markeredgecolor="none",
            label=cls_label(c),
        )
        for c in present
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(handles), 4),
        frameon=False,
        labelcolor="#c9d4e5",
        fontsize=9,
    )
    ax.axis("off")

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section">Neighborhood</div>', unsafe_allow_html=True)
        show_fig(fig)
        st.caption(
            "Colors show the true topic. The large outlined node is the selected paper."
        )
    with right:
        st.markdown(
            '<div class="section">Input given to the model</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="grid2">'
            + kpi("Papers", f"{len(subset)}", "nodes in the subgraph", small=True)
            + kpi(
                "Directed edges",
                f"{ei_used.shape[1]}",
                "both directions",
                small=True,
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{hops}-hop neighborhood, {keep}% of citations kept, seed {seed}.")
        with st.expander("Model output: ego graph (selected paper)"):
            st.json(sub_pred)
        with st.expander("Model output: full graph"):
            st.json(full)


# ----------------------------------------------------------------------------
# Tab 2: Model results
# ----------------------------------------------------------------------------
with tab_results:
    st.caption(
        "10 seeds, fixed public Planetoid split on Cora. Error bars show ± 1 std."
    )

    if not RESULTS_CSV.exists():
        st.error("artifacts/results.csv not found. Run `python -m src.train` first.")
    else:
        df = pd.read_csv(RESULTS_CSV)
        names = df["Model"].tolist()
        acc = np.array([parse_pm(s) for s in df["Accuracy"]])
        f1 = np.array([parse_pm(s) for s in df["Macro-F1"]])
        best = int(acc[:, 0].argmax())

        m1, m2, m3 = st.columns(3)
        stat_box(m1, names[best], "Best model")
        stat_box(m2, f"{acc[best, 0]:.3f}", "Best accuracy")
        if "MLP" in names:
            gain = (acc[best, 0] - acc[names.index("MLP"), 0]) * 100
            stat_box(m3, f"+{gain:.1f} pts", "Gain over MLP baseline")

        left, right = st.columns([3, 2])

        with left:
            st.markdown(
                '<div class="section">Accuracy and Macro-F1 by model</div>',
                unsafe_allow_html=True,
            )
            BLUE, TEAL, FG, GRID = "#4f8bf9", "#2dd4bf", "#e6edf3", "#2a3340"
            x = np.arange(len(names))
            w = 0.36

            fig2, ax2 = plt.subplots(figsize=(8, 4.6))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")

            for vals, offset, color, label in [
                (acc, -w / 2, BLUE, "Accuracy"),
                (f1, w / 2, TEAL, "Macro-F1"),
            ]:
                bars = ax2.bar(
                    x + offset,
                    vals[:, 0],
                    w,
                    color=color,
                    label=label,
                    edgecolor="none",
                    zorder=3,
                )
                ax2.errorbar(
                    x + offset,
                    vals[:, 0],
                    yerr=vals[:, 1],
                    fmt="none",
                    ecolor=FG,
                    elinewidth=1.2,
                    capsize=3,
                    capthick=1.2,
                    alpha=0.9,
                    zorder=4,
                )
                for bar, (mean, std) in zip(bars, vals):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        mean + std + 0.012,
                        f"{mean:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=FG,
                    )

            ax2.axvspan(best - 0.5, best + 0.5, color=BLUE, alpha=0.07, zorder=1)
            ax2.set_xticks(x, names, fontsize=11, color=FG)
            ax2.set_ylim(0, 1.0)
            ax2.set_yticks(np.arange(0, 1.01, 0.2))
            ax2.tick_params(axis="y", colors="#9aa7b8", labelsize=9, length=0)
            ax2.tick_params(axis="x", length=0, pad=8)
            ax2.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            ax2.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax2.spines[side].set_visible(False)
            ax2.spines["bottom"].set_color(GRID)
            ax2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=2,
                frameon=False,
                labelcolor=FG,
                fontsize=10,
            )
            show_fig(fig2)

        with right:
            st.markdown('<div class="section">Numbers</div>', unsafe_allow_html=True)
            st.dataframe(df, hide_index=True)
            st.markdown('<div class="section">Takeaways</div>', unsafe_allow_html=True)
            st.markdown(
                "- Graph models beat the feature-only MLP by about 23-25 points.\n"
                "- GCN, GAT and APPNP are within about one standard deviation of "
                "each other.\n"
                "- GAT was not tuned, so treat it as a baseline, not a verdict."
            )


# ----------------------------------------------------------------------------
# Tab 3: About
# ----------------------------------------------------------------------------
with tab_about:
    a1, a2 = st.columns([3, 2])
    with a1:
        st.markdown('<div class="section">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            "1. **Train**: PyTorch Geometric trains a two-layer GCN on Cora with early "
            "stopping on validation accuracy.\n"
            "2. **Export**: the best checkpoint is exported to ONNX with dynamic node "
            "and edge counts, then checked against PyTorch (max difference ~1e-6).\n"
            "3. **Serve**: this app loads the ONNX session once. The full-graph "
            "predictions are computed once and cached. For a selected paper, the app "
            "builds a neighborhood subgraph, normalizes the raw features and runs the "
            "model on that graph.\n"
            "4. **Explore**: change the depth, the share of citations kept and the seed "
            "to see how the prediction reacts to missing evidence."
        )
        st.code(
            "Streamlit UI  ->  ONNX Runtime (GCN)\n"
            "     |\n"
            "     +-- artifacts/cora.npz + artifacts/results.csv",
            language="text",
        )
    with a2:
        st.markdown('<div class="section">Stack</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                {
                    "Layer": ["Modeling", "Export", "Inference", "Interface"],
                    "Tools": [
                        "PyTorch, PyG",
                        "ONNX",
                        "ONNX Runtime",
                        "Streamlit",
                    ],
                }
            ),
            hide_index=True,
        )

    st.markdown('<div class="section">Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Results use the fixed public Planetoid split, so the ± values '
        "reflect seed variance only, not split variance. The subgraph path was "
        "checked on a prefix subgraph, not arbitrary ones. Dropping citations changes "
        "the graph the model sees, so ego-graph predictions are expected to differ "
        "from full-graph ones.</div>",
        unsafe_allow_html=True,
    )
