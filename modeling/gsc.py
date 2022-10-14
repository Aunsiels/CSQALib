from typing import Union
import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data, Batch
from .mlp import MLP
from utils.tensor_utils import make_one_hot

from transformers import BertModel


class GSCLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, node_emb, edge_index, edge_emb):
        aggr_out = self.propagate(edge_index, x=(
            node_emb, node_emb), edge_emb=edge_emb)  # [V, emb_dim]
        return aggr_out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class GSCMessagePassing(nn.Module):
    def __init__(
        self,
        hops: int,
        num_node_types: int,
        num_edge_types: int,
        hidden_size: int,
        emb_size: int = 1,
    ):
        super().__init__()
        self.num_nodetypes = num_node_types
        self.num_edgetypes = num_edge_types
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        self.edge_encoder = nn.Sequential(
            MLP(input_size=num_edge_types + num_node_types * 2,
                hidden_size=hidden_size,
                output_size=emb_size,
                num_layers=1,
                dropout_rate=0),
            nn.Sigmoid())
        self.hops = hops
        self.gnn_layers = nn.ModuleList([GSCLayer() for _ in range(hops)])

    def get_graph_edge_embedding(
        self,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        node_type: torch.LongTensor,
    ):
        # Prepare edge feature
        print(edge_type, self.num_edgetypes)
        edge_vec = make_one_hot(edge_type, self.num_edgetypes)  # [E, 39]
        head_type = node_type[edge_index[0]]  # [E,] #head=utils
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.num_nodetypes)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.num_nodetypes)  # [E,4]
        edge_embeddings = self.edge_encoder(
            torch.cat([edge_vec, head_vec, tail_vec], dim=1))  # [E, emb_size]
        return edge_embeddings

    def forward(self, graphs: Union[Data, Batch]):
        device = graphs.node_type.device
        num_nodes = graphs.node_type.shape[0]

        edge_embeddings = self.get_graph_edge_embedding(
            graphs.edge_index, graphs.edge_type, graphs.node_type)

        aggr_out = torch.zeros((num_nodes, 1)).to(device)
        for i in range(self.hops):
            # propagate and aggregate between nodes and edges
            aggr_out = self.gnn_layers[i](
                aggr_out, graphs.edge_index, edge_embeddings)
        return aggr_out  # [V, emb_size]


class GraphSoftCounter(nn.Module):
    def __init__(
        self,
        hops: int,
        num_node_types: int,
        num_edge_types: int,
        hidden_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.message_passing = GSCMessagePassing(
            hops, num_node_types, num_edge_types, hidden_size)

    def forward(self, graphs: Batch, lm_context: torch.Tensor):
        ctx_nodes_idx = graphs.ptr[:-1]
        graph_emb = self.message_passing(graphs)[ctx_nodes_idx]  # [batch_size, 1]
        return graph_emb


class LMGSC(nn.Module):
    def __init__(
        self,
        hops: int,
        num_node_types: int,
        num_edge_types: int,
        gsc_hidden_size: int,

        lm: BertModel,
    ):
        super().__init__()
        node_emb_size = 1

        self.gnn = GSCMessagePassing(
            hops, num_node_types, num_edge_types, gsc_hidden_size, node_emb_size)
        self.ctx_scorer = nn.Linear(lm.config.hidden_size, 1)
        self.graph_scorer = MLP(input_size=node_emb_size,
                                hidden_size=32,
                                output_size=1,
                                num_layers=1,
                                dropout_rate=0.2)

    def forward(self, graphs: Batch, lm_context):
        ctx_nodes_idx = graphs.ptr[:-1]
        graph_emb = self.gnn(graphs)[ctx_nodes_idx]  # [batch, node_emb]
        context_score = self.ctx_scorer(lm_context)
        graph_score = self.graph_scorer(graph_emb)
        qa_score = context_score + graph_score
        return qa_score
