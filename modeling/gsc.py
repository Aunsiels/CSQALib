import torch
from torch import nn
import numpy as np


class GraphHardCounter(nn.Module):
    def __init__(self, num_rels):
        super().__init__()
        self.hidden_size = 1
        # 3 node types
        self.scorer = nn.Embedding(num_rels * 3 * 3, 1)

    def forward(self, graphs, text):
        def score(graph):
            device = graph.node_type.device
            nt = graph.node_type.detach().cpu().numpy()
            et = graph.edge_type.detach().cpu().numpy()
            ei = graph.edge_index.T.detach().cpu().numpy()

            edge_enc = torch.tensor(np.array([
                et[i] * 3*3 + nt[ei[i, 0]] * 3 + nt[ei[i, 1]]
                for i in range(et.shape[0])
            ])).to(device)
            return [self.scorer(edge_enc).sum()]

        return torch.stack([score(graph) for graph in graphs.to_data_list()])
