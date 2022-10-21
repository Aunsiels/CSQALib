from modeling.gsc import GraphSoftCounter
from modeling.lm_gnn import EmptyGnn
from modeling.qagnn import QAGNN
from parameters import ARGS


def load_gnn(gnn_name, lm_dim, hid_dim, num_rels):
    if gnn_name == "none":
        return EmptyGnn()
    if gnn_name == "gsc":
        return GraphSoftCounter(hops=ARGS.hops,
                                num_node_types=4,
                                num_edge_types=num_rels,
                                hidden_size=hid_dim)
    if gnn_name == "qagnn":
        return QAGNN(lm_dim, hid_dim, num_rels, ARGS.hops)
    raise NotImplementedError
