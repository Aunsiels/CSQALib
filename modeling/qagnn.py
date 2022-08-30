import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.data import Data as Graph
from torch_geometric.data.batch import Batch as Graphs


def add_ctx_qagnn(graph: Graph, num_rels, ctx_node_emb: torch.Tensor):
    assert set(graph.keys) >= {'node_type', 'edge_index',
                               'node_emb', 'edge_type'}
    # prepend new node: no embedding, type 3
    device = ctx_node_emb.device
    ctx_node_type = torch.tensor([3], dtype=torch.long).to(device)

    node_emb = torch.cat((ctx_node_emb.view(1,-1), graph.node_emb), 0)
    node_type = torch.cat((ctx_node_type, graph.node_type), 0)

    # add edges between context and nodes with types 0 and 1
    (qnodes,) = torch.where(node_type == 0)
    (anodes,) = torch.where(node_type == 1)
    rq, ra, irq, ira = num_rels + 0, num_rels + 1, num_rels + 2, num_rels + 3
    new_edges = torch.tensor(
        [[0, q, rq] for q in qnodes]
        + [[0, a, ra] for a in anodes]
        + [[q, 0, irq] for q in qnodes]
        + [[a, 0, ira] for a in anodes],
        dtype=torch.long,
    ).view(-1,3).t().to(device)

    edge_index = torch.cat((new_edges[:2], graph.edge_index+1), 1)
    edge_type = torch.cat((new_edges[2], graph.edge_type), 0)

    return Graph(
        node_type=node_type,
        node_emb=node_emb,
        edge_index=edge_index,
        edge_type=edge_type,
    )


class QAGNN(nn.Module):
    def __init__(
        self,
        lm_dim,
        hid_dim,
        num_rels,
        num_hops,
    ):
        super().__init__()
        self.lm_dim = lm_dim
        self.hid_dim = hid_dim
        self.num_rels = num_rels
        self.num_hops = num_hops

        self.embed_lm = nn.Linear(lm_dim, hid_dim)
        self.relev_emb = nn.Bilinear(lm_dim, hid_dim, hid_dim)
        self.edge_emb = nn.Embedding(num_rels + 4, hid_dim)
        self.gats = nn.ModuleList([gnn.GATConv(hid_dim, hid_dim, edge_dim=hid_dim) for _ in range(num_hops)])

    def forward(
        self,
        graphs: Graphs,
        lm_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Node 0 of each sub graph is the context.
        other nodes are concepts, edges are relations.

        node_type: int[nodes]
            values (0: in question, 1: in answer, 2: from subgraph, 3: qa_context)
        node_emb: float[nodes, dim_nodes]
        edge_index: int[2, edges]
            2 rows (from, to)
            in range nodes
        edge_type: int[edges]
            in range rel_types
        lm_context: float[batch_size, lm_dim]
        """
        # set context nodes embedding
        ctx_nodes = graphs.ptr[:-1]
        ctx_emb = self.embed_lm(lm_context)
        graphs = Graphs.from_data_list(
            [add_ctx_qagnn(data, self.num_rels, ctx)
             for data, ctx in zip(graphs.to_data_list(), ctx_emb)]
        ).to(lm_context.device)

        node_index = graphs.batch

        nodes = self.relev_emb(
            lm_context[node_index],
            graphs.node_emb,
        )
        edges = self.edge_emb(graphs.edge_type)

        for gat in self.gats:
            nodes = gat(nodes, graphs.edge_index, edges)
            nodes = F.gelu(nodes)
            # TODO edge update ?

        return nodes[ctx_nodes]
