import json
import logging
import sys
from time import time

import pandas as pd
import wandb

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset as HFDataset
from transformers import AutoModel, AutoTokenizer

from src.build_graph import build_graph_relidx
from src.extract_graph import extract_graph
from src.matcher import Matcher
from src.ranker import Ranker
from src.datasets import (
    BatchDataset, ZipDataset,
    BatchCollator, ZipCollator,
    TextCollator, GraphCollator,
)
from loaders import (
    load_edges,
    load_rels,
    load_embedder,
    load_qa,
)

from modeling.lm_gnn import LM_GNN
from modeling.qagnn import QAGNN


def train(
    kb_name="cpnet_en",
    emb_name="glove",
    qa_task="csqa",
    lm_name="roberta-large",
    gnn_name="qagnn",
    batch_size=16,
    epochs=20,
    learning_rate=5e-5,
    top_k=100,
    num_hops=3,
):
    wandb.init(
        project="csqa",
        entity="mhslr",
        config=dict(
            kb_name=kb_name,
            emb_name=emb_name,
            qa_task=qa_task,
            lm_name=lm_name,
            gnn_name=gnn_name,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            top_k=top_k,
            num_hops=num_hops,
        ),
    )
    t0 = time()
    edges = load_edges(kb_name)
    rels = load_rels(kb_name)
    embedder = load_embedder(emb_name)
    train_rec, test_rec = load_qa(qa_task)
    logging.info('123')
    print('loaded %.2fs' % (time() - t0))

    graph, idx2rel = build_graph_relidx(edges, rels)
    matcher = Matcher(list(graph.nodes))
    ranker = Ranker()
    print('graph matcher ranker %.2fs' % (time() - t0))

    num_choices = len(train_rec[0]["choices"])
    num_rels = len(idx2rel)

    tokenizer = AutoTokenizer.from_pretrained(lm_name)

    def get_dataloader(records):
        # flattened & graph
        df = (
            pd.DataFrame.from_records(records)
            .explode("choices")
            .rename({"choices": "answer"}, axis=1)
            .to_records()
        )
        graph = [extract_graph(x.question, x.answer, matcher, graph,
                               ranker, embedder, top_k) for x in df.itertuples()]
        text = HFDataset.from_pandas(graph)

        dataset = BatchDataset(ZipDataset([text, graph]), num_choices)
        collator = BatchCollator(ZipCollator(
            [TextCollator(tokenizer), GraphCollator]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collator)
        return dataloader

    train_dataloader = get_dataloader(train_rec)
    test_dataloader = get_dataloader(test_rec)
    print('dataloaders %.2fs' % (time() - t0))


    language_model = AutoModel.from_pretrained(lm_name)
    lm_dim = language_model.config.hidden_size

    if gnn_name == "qagnn":
        gnn = QAGNN(lm_dim, embedder.dim, num_rels, num_hops)  # TODO
    else:
        raise NotImplementedError

    model = LM_GNN(language_model, lm_dim, gnn, gnn.hid_dim)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print('start training %.2fs' % (time() - t0))

    for epoch in range(epochs):
        for text, graph in train_dataloader:
            logits = model(text, graph)
            labels = text['labels'][::num_choices]  # unbatching
            loss = F.cross_entropy(logits.view(-1, 5), labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

if __name__ == "__main__":
    cfg = {}
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
        cfg = json.load(open(cfg_path))
    train(**cfg)