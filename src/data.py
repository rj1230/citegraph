import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def load_cora(root="data/Planetoid"):
    dataset = Planetoid(root=root, name="Cora", transform=T.NormalizeFeatures())
    return dataset, dataset[0]
