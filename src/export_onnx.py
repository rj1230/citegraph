import numpy as np
import torch
import onnxruntime as ort

from src.data import load_cora
from src.models import SimpleGCN


def main(weights="artifacts/gcn.pt", out_path="artifacts/simple_gcn_cora.onnx"):
    dataset, data = load_cora()
    model = SimpleGCN(data.num_features, 64, dataset.num_classes)
    model.load_state_dict(torch.load(weights))
    model.eval()

    torch.onnx.export(
        model,
        (data.x, data.edge_index),
        out_path,
        dynamo=False,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["node_features", "edge_indices"],
        output_names=["logits"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            "edge_indices": {1: "num_edges"},
        },
    )
    print("Exported", out_path)

    sess = ort.InferenceSession(out_path)
    feed = lambda x, ei: {"node_features": x.numpy(), "edge_indices": ei.numpy()}

    # Full graph
    with torch.no_grad():
        ref = model(data.x, data.edge_index).numpy()
    out = sess.run(None, feed(data.x, data.edge_index))[0]
    print("Full graph max diff:", np.abs(out - ref).max())

    # 1000-node subgraph (tests dynamic_axes)
    n = 1000
    mask = (data.edge_index[0] < n) & (data.edge_index[1] < n)
    x_s, ei_s = data.x[:n], data.edge_index[:, mask]
    with torch.no_grad():
        ref_s = model(x_s, ei_s).numpy()
    out_s = sess.run(None, feed(x_s, ei_s))[0]
    print("Subgraph max diff:", np.abs(out_s - ref_s).max())


if __name__ == "__main__":
    main()
