import torch_geometric.nn as gnn

import torch
from torch import nn
import torch_geometric.nn as gnn
import torch.nn.functional as F



class MHGRN(nn.Module):
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
        node_types = 3

        self.node_type_emb = nn.Embedding(node_types, hid_dim)
        self.W = nn.Parameter(torch.randn(
            (num_hops, num_rels, hid_dim, hid_dim)))
        # t -> rel_transition_weight
        # delta -> rel_lm_weight
        # f -> src_lm_weight
        # g -> dst_lm_weight
        self.rel_transition_weight = nn.Parameter(
            torch.randn((num_rels, num_rels)))
        self.rel_lm_weight = [nn.Linear(lm_dim, 1) for _ in range(num_rels)]
        self.src_lm_weight = [nn.Linear(lm_dim, 1) for _ in range(node_types)]
        self.dst_lm_weight = [nn.Linear(lm_dim, 1) for _ in range(node_types)]
        self.lm_z = nn.Bilinear(lm_dim, hid_dim, 1)
        self.V = nn.Linear(lm_dim+hid_dim, hid_dim)

    def forward(
        self,
        graphs,
        lm_context,
    ):
        """
        All nodes are concepts.

        node_type: int[nodes]
            values (0: in question, 1: in answer, 2: from subgraph)
        node_emb: float[nodes, dim_nodes]
        edge_index: int[2, edges]
            2 rows (from, to)
            in range nodes
        edge_type: int[edges]
            in range rel_types
        """
        node_index = graphs.batch
        node_type = graphs.node_type
        node_emb = graphs.node_emb
        edge_index = graphs.edge_index
        edge_type = graphs.edge_type
        num_nodes = node_type.shape[0]
        edges_for_dfs = [
            list(
                zip(
                    edge_index[1][edge_index[0] ==
                                  src], edge_type[edge_index[0] == src]
                )
            )
            for src in range(num_nodes)
        ]

        def dfs(depth: int, cur: int, path: tuple):
            if path:
                yield path
            if depth == 0:
                return
            for nxt, rel in edges_for_dfs[cur]:
                yield from dfs(depth - 1, nxt, path + (rel, nxt))

        def compute_act(i, j, rels, s):
            """CRF ~ beta(r1,...,rk,s) * gamma(i,j,s)
            used to softmax paths of same length"""
            # print(i,j,rels, s.shape)
            return torch.exp(
                self.src_lm_weight[node_type[i]](s)
                + self.dst_lm_weight[node_type[j]](s)
                + sum(self.rel_lm_weight[r](s) for r in rels)
                + sum(
                    self.rel_transition_weight[r1, r2] for r1, r2 in zip(rels, rels[1:])
                )
            ).view(())

        def compute_zak(i, path, s):
            """
            returns
              change in z[i,k]
              activation
              k length of path
            """
            # path is [rel1, node1, rel2, node2, ...]
            j = path[-1]
            rel_types = path[::2][::-1]  # reversed

            padded_rel_types = rel_types + \
                (0,) * (self.num_hops - len(rel_types))
            # r1 .. rk 0 .. 0 (to reach rK)
            z = node_emb[j]
            for rank, rtype in enumerate(padded_rel_types):
                z = self.W[rank, rtype] @ z
            act = compute_act(i, j, padded_rel_types, s)
            return z, act, len(rel_types)

        # main loop
        # TODO what's h ???
        z2 = torch.zeros((num_nodes, self.hid_dim)).to(node_emb.device)
        for i in range(num_nodes):
            s = lm_context[node_index[i]]
            zak = [compute_zak(i, p, s) for p in dfs(self.num_hops-1, i, ())]
            if not zak:
                continue
            z,a,k = map(list, zip(*zak))
            z = torch.stack(z, axis=0)
            a = torch.tensor(a)
            k = torch.tensor(k, dtype=torch.long)

            z1 = torch.zeros((self.num_hops, self.hid_dim))
            for kk in range(self.num_hops):
                if (k == kk).any():
                    z1[kk] = (z[k == kk] * a[k == kk].view(-1,1)).sum(0) / a[k == kk].sum()
            weights = self.lm_z(s.expand(self.num_hops, -1), z1)
            z2[i] = (F.softmax(weights, 0) * z1).sum(axis=0)
        h2 = self.V(torch.cat((lm_context[node_index], z2), 1))
        return gnn.global_max_pool(h2, graphs.batch)
