from typing import List, Tuple
import pandas as pd
import networkx as nx

NODES_BLACKLIST = {"uk", "us", "take", "make", "object", "person", "people"}


def get_fwd_bwd_types(row):
    fwd = row.target
    bwd = fwd if row.symmetric else '~' + fwd
    if row.reversed:
        fwd, bwd = bwd, fwd
    return fwd, bwd


def build_relmap(relations: pd.DataFrame):
    return {row.source: get_fwd_bwd_types(row)
            for row in relations.itertuples()
            if row.keep}


def get_fwd_bwd(row, relmap):
    fwd, bwd = relmap[row.Type]
    yield fwd, row.Head, row.Tail, row.Weight
    yield bwd, row.Tail, row.Head, row.Weight


def build_graph(edges: pd.DataFrame, relations: pd.DataFrame, use_idx: bool) -> Tuple[nx.MultiDiGraph, List[str]]:
    relmap = build_relmap(relations)
    rel_idx = {}

    graph = nx.MultiDiGraph()
    edges2 = []
    for _, row in edges.iterrows():
        if row.Type in relmap and row.Head not in NODES_BLACKLIST and row.Tail not in NODES_BLACKLIST:
            for Rel, head, tail, Weight in get_fwd_bwd(row, relmap):
                edges2.append((Rel, head, tail, Weight))

    edges2 = list(set(edges2))  # unique
    if use_idx:
        edges3 = []
        for Rel, head, tail, Weight in edges2:
            if Rel not in rel_idx:
                rel_idx[Rel] = len(rel_idx)
            edges3.append((rel_idx[Rel], head, tail, Weight))
        edges2 = edges3

    for Rel, head, tail, Weight in edges2:
        graph.add_edge(head, tail, Rel=Rel, Weight=Weight)

    return graph, list(rel_idx)


def build_graph_with_relation_idx(edges: pd.DataFrame, relations: pd.DataFrame):
    return build_graph(edges, relations, True)
