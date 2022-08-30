import torch
import torch.nn.functional as F
from torch.optim import SGD

from modeling.gsc import GraphHardCounter
from torch_geometric.data import Data as Graph, Batch as Graphs


def test_ghc():
    num_rels = 17
    V, E = 3, 5
    graph = Graph(
        node_type=torch.randint(0, 3, (V, 1)),
        edge_type=torch.randint(0, num_rels, (E, 1)),
        edge_index=torch.tensor([
            [0, 1],
            [0, 2],
            [1, 2],
            [2, 1],
            [0, 1],
        ], dtype=torch.int).T,
    )
    graphs = Graphs.from_data_list([graph]*3)

    model = GraphHardCounter(num_rels)
    optim = SGD(model.parameters(), lr=1e-3)
    for _ in range(100):
        out = model(graphs, ...)
        loss = F.mse_loss(out.view(-1), torch.tensor([1.]))
        
        model.zero_grad()
        loss.backward()
        optim.step()

        print(loss.item())