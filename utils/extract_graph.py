import numpy as np
import networkx as nx
import torch
from torch_geometric.data.data import Data


from utils.concept_embedder import Embedder
from utils.matcher import Matcher
from utils.ranker import CosineSimilarityRanker, get_top_k


def get_2_hops_subgraph(graph: nx.MultiDiGraph, src, dst) -> nx.MultiDiGraph:
    src1 = set(i for x in src for i in graph[x])
    dst1 = set(i for x in dst for i in graph[x])
    nodes = set(src) | set(dst) | (src1 & dst1)
    return graph.subgraph(list(nodes))


def extract_graph(question: str, answer: str, matcher: Matcher, graph: nx.MultiDiGraph,
                  ranker: CosineSimilarityRanker, embedder: Embedder, top_k):
    qcids = matcher.match_one(question)
    acids = matcher.match_one(answer)
    qcids -= acids
    # NB: acids could be empty...

    subgraph1 = get_2_hops_subgraph(graph, qcids, acids)
    cids1 = list(subgraph1.nodes)
    boost = [cid in qcids or cid in acids for cid in cids1]
    scores = ranker(question, cids1) + 2 * np.array(boost)

    cids2 = get_top_k(cids1, scores, top_k)
    subgraph2 = subgraph1.subgraph(cids2)
    embeddings = embedder(cids2)
    return as_data(cids2, subgraph2, qcids, acids, embeddings)


def as_data(cids, graph: nx.MultiDiGraph, qcids, acids, embeddings: np.array):
    cid_to_id2 = {cid: i for i, cid in enumerate(cids)}
    return Data(
        node_emb=torch.tensor(embeddings, dtype=torch.float32),
        node_type=torch.tensor(
            [0 if cid in acids else 1 if cid in qcids else 2 for cid in cids],
            dtype=torch.long),
        edge_index=torch.tensor([[cid_to_id2[u], cid_to_id2[v]]
                                 for u, v, k in graph.edges], dtype=torch.long).T,
        edge_type=torch.tensor([graph.get_edge_data(u, v, k)["Rel"]
                                for u, v, k in graph.edges], dtype=torch.long)
    )


def add_context_node(graph: Data, num_rels, ctx_node_emb: torch.Tensor = None) -> Data:
    """
    inserts a node with type 3
    adds 4 types of relations
    if ctx_node_emb is None: ctx_node_emb = zeros(...)
    """
    assert set(graph.keys) >= {'node_type', 'edge_index',
                               'node_emb', 'edge_type'}
    device = graph.node_emb.device

    if ctx_node_emb is None:
        ctx_node_emb = torch.zeros_like(graph.node_emb[0], device=device)
    ctx_node_type = torch.tensor([3], dtype=torch.long).to(device)

    node_emb = torch.cat((ctx_node_emb.view(1, -1), graph.node_emb), 0)
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
    ).view(-1, 3).t().to(device)

    edge_index = torch.cat((new_edges[:2], graph.edge_index+1), 1)
    edge_type = torch.cat((new_edges[2], graph.edge_type), 0)

    return Data(
        node_type=node_type,
        node_emb=node_emb,
        edge_index=edge_index,
        edge_type=edge_type,
    )
