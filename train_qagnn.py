import json
import sys
from tqdm import tqdm

import pandas as pd
import wandb

import torch
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

from modeling.lm_gnn import LM_GNN, EmptyGnn
from modeling.qagnn import QAGNN
from modeling.gsc import GraphHardCounter


def load_gnn(gnn_name, lm_dim, hid_dim, num_rels, num_hops):
    if gnn_name == "none":
        return EmptyGnn()
    if gnn_name == "gsc":
        return GraphHardCounter(num_rels)
    if gnn_name == "qagnn":
        return QAGNN(lm_dim, hid_dim, num_rels, num_hops)
    raise NotImplementedError


def train(
    kb_name="cpnet_en",
    emb_name="glove",
    qa_task="csqa",
    lm_name="roberta-large",
    gnn_name="gsc",
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
    edges = load_edges(kb_name)
    rels = load_rels(kb_name)
    embedder = load_embedder(emb_name)
    train_rec, test_rec = load_qa(qa_task)

    knowledge_graph, idx2rel = build_graph_relidx(edges, rels)
    matcher = Matcher(list(knowledge_graph.nodes))
    ranker = Ranker()

    num_choices = len(train_rec[0]["choices"])
    num_rels = len(idx2rel)

    tokenizer = AutoTokenizer.from_pretrained(lm_name)

    def get_dataloader(records):
        # flattened & graph
        df = (
            pd.DataFrame.from_records(records)
            .explode("choices")
            .rename({"choices": "answer"}, axis=1)
        )
        graphs = [extract_graph(x.question, x.answer, matcher, knowledge_graph,
                                ranker, embedder, top_k) for x in tqdm(df.itertuples(), desc='dataloader', total=len(df))]
        text = HFDataset.from_pandas(df)

        dataset = BatchDataset(ZipDataset([text, graphs]), num_choices)
        collator = BatchCollator(ZipCollator(
            [TextCollator(tokenizer), GraphCollator]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collator)
        return dataloader

    train_dataloader = get_dataloader(train_rec)
    test_dataloader = get_dataloader(test_rec)

    language_model = AutoModel.from_pretrained(lm_name)
    lm_dim = language_model.config.hidden_size

    gnn = load_gnn(gnn_name, lm_dim, embedder.dim, num_rels, num_hops)
    model = LM_GNN(language_model, lm_dim, gnn, gnn.hid_dim)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(epochs):
        wandb.log({"epoch": epoch})
        for text, graph in train_dataloader:
            text.to(device), graph.to(device)
            labels = text.pop('labels')[::num_choices]  # unbatching
            logits = model(text, graph)
            loss = F.cross_entropy(logits.view(-1, 5), labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item()})

        with torch.no_grad():
            correct = 0
            for text, graph in test_dataloader:
                text.to(device), graph.to(device)
                labels = text.pop('labels')[::num_choices]  # unbatching
                logits = model(text, graph)
                answer = logits.view(-1, 5).argmax(1)
                correct += (answer == labels).sum().item()
            wandb.log({"validation/accuracy": correct /
                      len(test_dataloader.dataset)})


if __name__ == "__main__":
    cfg = {}
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
        cfg = json.load(open(cfg_path))
    train(**cfg)
