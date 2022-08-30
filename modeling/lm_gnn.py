from turtle import forward
from typing import Dict
import torch
from torch import nn
from torch_geometric.data.batch import Batch as Graphs

from transformers import BertModel, BertConfig
from transformers.modeling_outputs import BaseModelOutput


class GNN(nn.Module):
    def __init__(self):
        super().init()
        self.hidden_size = 0

    def forward(self, graphs: Graphs, lm_context: torch.Tensor):
        raise NotImplementedError


class EmptyGnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 0

    def forward(self, graphs: Graphs, lm_context: torch.Tensor):
        return torch.Tensor(size=(lm_context.shape[0], 0), device=lm_context.device)


class LM_GNN(nn.Module):
    def __init__(
        self,
        lm: BertModel,
        gnn: GNN,
    ):
        super().__init__()
        self.lm = lm
        # https://huggingface.co/docs/transformers/v4.21.2/en/model_doc/bert#transformers.BertConfig
        lm_dim = lm.config.hidden_size

        self.gnn = gnn
        gnn_dim = gnn.hidden_size

        self.clf = nn.Sequential(
            nn.Linear(lm_dim+gnn_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, tokens: Dict[str, torch.Tensor], graphs: Graphs) -> torch.Tensor:
        output: BaseModelOutput = self.lm(**tokens)
        lm_context = output.last_hidden_state[:, 0]  # embedding of [CLS] token
        graph_context = self.gnn(graphs, lm_context)
        return self.clf(torch.cat((lm_context, graph_context), 1))
