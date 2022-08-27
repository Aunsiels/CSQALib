import numpy as np
import networkx as nx
import torch
from torch_geometric.data.data import Data

from .concept_embedder import Embedder
from .matcher import Matcher
from .ranker import Ranker, get_top_k


def get_2_hops_subgraph(graph: nx.MultiDiGraph, src, dst) -> nx.MultiDiGraph:
    src1 = set(i for x in src for i in graph[x])
    dst1 = set(i for x in dst for i in graph[x])
    nodes = set(src) | set(dst) | (src1 & dst1)
    return graph.subgraph(list(nodes))


def extract_graph(question: str, answer: str, matcher: Matcher, graph: nx.MultiDiGraph, ranker: Ranker, embedder: Embedder, top_k):
    qcids = matcher(question)
    acids = matcher(answer)
    qcids -= acids
    # NB: acids could be empty...

    subgraph1 = get_2_hops_subgraph(graph, qcids, acids)
    cids1 = list(subgraph1.nodes)
    boost = [cid in qcids or cid in acids for cid in cids1]
    scores = ranker(cids1) + 2 * np.array(boost)

    cids2 = get_top_k(cids1, scores, top_k)
    subgraph2 = subgraph1.subgraph(cids2)
    embeddings = embedder(cids2)
    return as_data(cids2, subgraph2, qcids, acids, embeddings)


def as_data(cids, graph: nx.MultiDiGraph, qcids, acids, embeddings: np.array):
    cid_to_id2 = {cid: i for i, cid in enumerate(cids)}
    return Data(
        x=torch.Tensor(embeddings),
        node_type=torch.Tensor(
            [0 if cid in acids else 1 if cid in qcids else 2 for cid in cids]),
        edge_index=torch.Tensor([[cid_to_id2[u], cid_to_id2[v]]
                                 for u, v, k in graph.edges]).T,
        edge_type=torch.Tensor([graph.get_edge_data(u, v, k)["Rel"]
                                for u, v, k in graph.edges])
    )
