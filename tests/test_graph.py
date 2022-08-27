import pandas as pd
from src.build_graph import build_graph


def test_graph():
    edges = pd.DataFrame.from_records([
        ("motivatedbygoal", "love", "money", 1.),
        ("antonym", "love", "hate", 1.),
        ("hascontext", "sail", "boat", 1.),  # hascontext has keep = False
        ("isa", "us", "country", 1.),  # in NODE_BLACKLIST
    ], columns=["Type", "Head", "Tail", "Weight"])
    relations = pd.read_csv(
        '/home/mh/lix/csqa/v4/data/cpnet-rels.tsv', sep='\t')
    graph = build_graph(edges, relations)
    assert set(graph.nodes) == {'love', 'money', 'hate'}
    assert graph["love"]["money"][0]["Rel"] == "~causes"
    assert graph["money"]["love"][0]["Rel"] == "causes"