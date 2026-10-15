import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_cora
from src.models import SimpleGCN, MLP, GAT, APPNPNet


def run(make_model, data, lr, wd, seed, epochs=300, patience=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = make_model()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, bad = 0.0, None, 0
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        out = m(data.x, data.edge_index)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(data.x, data.edge_index).argmax(1)
        val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if val > best_val:
            best_val, best_state, bad = val, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        pred = m(data.x, data.edge_index).argmax(1)
    yt, pt = data.y[data.test_mask].numpy(), pred[data.test_mask].numpy()
    return m, accuracy_score(yt, pt), f1_score(yt, pt, average="macro")


def main(seeds=10, save_gcn=True):
    dataset, data = load_cora()
    F_, C_ = data.num_features, dataset.num_classes
    configs = {
        "MLP": (lambda: MLP(F_, 64, C_), 0.01, 5e-4),
        "GCN": (lambda: SimpleGCN(F_, 64, C_), 0.01, 5e-4),
        "GAT": (lambda: GAT(F_, C_), 0.005, 5e-4),
        "APPNP": (lambda: APPNPNet(F_, 64, C_), 0.01, 5e-4),
    }
    rows = []
    for name, (make, lr, wd) in configs.items():
        res = np.array([run(make, data, lr, wd, s)[1:] for s in range(seeds)])
        rows.append(
            {
                "Model": name,
                "Accuracy": f"{res[:, 0].mean():.3f} ± {res[:, 0].std():.3f}",
                "Macro-F1": f"{res[:, 1].mean():.3f} ± {res[:, 1].std():.3f}",
            }
        )
        print(rows[-1])
    pd.DataFrame(rows).to_csv("artifacts/results.csv", index=False)

    if save_gcn:
        m, acc, f1 = run(configs["GCN"][0], data, 0.01, 5e-4, seed=42)
        torch.save(m.state_dict(), "artifacts/gcn.pt")
        print(f"Saved artifacts/gcn.pt (seed 42) acc={acc:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    main()
