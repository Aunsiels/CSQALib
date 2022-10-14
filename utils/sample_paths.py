import random
import networkx as nx
from tqdm import tqdm


def sample_paths(graph: nx.MultiDiGraph, start_nodes=None, num_paths=36000):
    """sampling paths [(e1, r1, e2, .., rk, ek+1)]
    - distinct tuples
    - paths are simple (nodes are distinct)
    - rels on each path are distinct
    """
    if not start_nodes:
        start_nodes = list(graph.nodes)

    MAX_PATHS_BY_LEN = [(1, 10), (2, 8), (3, 6)]
    ATTEMPTS_PATH = 3
    ATTEMPTS_NODE = 3
    ATTEMPTS_REL = 3

    def sample_path(start_node, path_len):
        seen_nodes = {start_node}
        seen_rels = set()

        path = [start_node]
        cur_node = start_node
        for _ in range(path_len):

            rel, next_node = next(
                (
                    (rel, next_node)
                    for next_node in (
                        random.choice(list(graph[cur_node])) for _ in range(ATTEMPTS_NODE)
                    )
                    if next_node not in seen_nodes
                    for rel in (
                        random.choice(graph[cur_node][next_node])['Rel']
                        for _ in range(ATTEMPTS_REL)
                    )
                    if rel not in seen_rels
                ),
                (None, None),
            )

            if rel is not None:
                path.extend((rel, next_node))
                seen_nodes.add(next_node)
                seen_rels.add(rel)
                cur_node = next_node
            else:
                return None
        return tuple(path)

    seen_paths = set()
    for start_node in tqdm(range(num_paths), 'sampling paths'):
        start_node = random.choice(start_nodes)
        for path_len, nmax in MAX_PATHS_BY_LEN:
            for _ in range(nmax):
                path = next(
                    (
                        p
                        for p in (
                            sample_path(start_node, path_len)
                            for _ in range(ATTEMPTS_PATH)
                        )
                        if p and p not in seen_paths
                    ),
                    None,
                )
                if path:
                    seen_paths.add(path)

    return list(seen_paths)
