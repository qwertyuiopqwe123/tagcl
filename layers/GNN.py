import torch
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, APPNP
import torch.nn as nn
import torch.nn.functional as F


# Borrowed from BGRL
class GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm_mm=0.99, flag=1):
        super().__init__()

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.flag = flag

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if self.flag:  # 使用 GCNConv
                layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'), )
            else:  # 使用 MLP
                layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'), )

            layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            layers.append(nn.PReLU())

        self.model_gcn = Sequential('x, edge_index', layers) if self.flag else None
        self.model_gcnt = Sequential('x', layers) if not self.flag else None

    def forward(self, data):
        if self.flag:
            return self.model_gcn(data.x, data.edge_index)
        else:
            return self.model_gcnt(data.x, data.edge_index)

    def reset_parameters(self):
        if self.flag:
            self.model_gcn.reset_parameters()
        else:
            self.model_gcnt.reset_parameters()
