from typing import Dict
import torch
from torch import nn
from torch_geometric.data.batch import Batch as Graphs


class EmptyGnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid_dim = 0

    def forward(self, graphs: Graphs, lm_context: torch.Tensor):
        return torch.Tensor(size=(lm_context.shape[0], 0), device=lm_context.device)


class LM_GNN(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        lm_dim: int,
        gnn: nn.Module,
        gnn_dim: int,
    ):
        super().__init__()
        self.lm = lm
        self.gnn = gnn
        self.clf = nn.Linear(lm_dim+gnn_dim, 1)

    def forward(self, tokens: Dict[str, torch.Tensor], graphs: Graphs) -> torch.Tensor:
        lm_context = self.lm(**tokens).pooler_output
        graph_context = self.gnn(graphs, lm_context)
        return self.clf(torch.cat((lm_context, graph_context), 1))
