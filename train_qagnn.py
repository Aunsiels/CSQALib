from datetime import datetime
import shelve
from tqdm import tqdm

import pandas as pd
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset as HFDataset
from transformers import AutoModel, AutoTokenizer

from modeling.gnn_loader import load_gnn
from parameters import ARGS
from utils.build_graph import build_graph_with_relation_idx
from utils.extract_graph import extract_graph
from utils.matcher import Matcher
from utils.ranker import CosineSimilarityRanker
from utils.datasets import (
    BatchDataset, ZipDataset,
    BatchCollator, ZipCollator,
    TextCollator, GraphCollator,
)
from loaders.loader_factory import load_edges, load_relations, load_embedder, load_qa

from modeling.lm_gnn import LM_GNN


def get_cached(key, maxsize=10):
    if not isinstance(key, str):
        key = str(key)  # simple suboptimal
    with shelve.open("data/shelve") as db:
        ts = datetime.now().isoformat()
        if "_access" not in db:
            db["_access"] = {}
        access_logs = db["_access"]
        access_logs[key] = ts

        by_decr_time = sorted(
            [(ts1, key1) for key1, ts1 in access_logs.items()], reverse=True)
        for ts1, key1 in by_decr_time[maxsize:]:
            if key1 in db:
                del key1

        db["_access"] = access_logs

        if key not in db:
            db[key] = load_dataset()
        return db[key]


def load_dataset():
    edges = load_edges(ARGS.kb_type, ARGS.kb_path)
    relations = load_relations(ARGS.rel_path)
    embedder = load_embedder(ARGS.node_embeddings)
    train_rec, test_rec = load_qa(ARGS.train_file, ARGS.test_file)

    knowledge_graph, idx2rel = build_graph_with_relation_idx(edges, relations)
    matcher = Matcher(list(knowledge_graph.nodes))
    ranker = CosineSimilarityRanker()

    num_choices = len(train_rec[0]["choices"])
    num_rels = len(idx2rel)
    embed_dim = embedder.dim

    def get_dataset(records):
        # flattened
        df = (
            pd.DataFrame.from_records(records)
            .explode("choices")
            .rename({"choices": "answer"}, axis=1)
        )

        graphs = [extract_graph(x.question, x.answer, matcher, knowledge_graph, ranker, embedder, ARGS.top_k)
                  for x in tqdm(df.itertuples(), desc='Extracting sub-graphs', total=len(df))]
        text = HFDataset.from_pandas(df)

        dataset = BatchDataset(ZipDataset([text, graphs]), num_choices)
        return dataset

    train_dataset = get_dataset(train_rec)
    test_dataset = get_dataset(test_rec)
    return (
        num_choices,
        num_rels,
        embed_dim,
        train_dataset,
        test_dataset,
    )


def train():
    key_cache = (ARGS.kb_type, ARGS.rel_path, ARGS.kb_path, ARGS.node_embeddings,
                 ARGS.train_file, ARGS.test_file, ARGS.top_k)
    num_choices, num_rels, embed_dim, train_dataset, test_dataset = get_cached(key_cache)

    tokenizer = AutoTokenizer.from_pretrained(ARGS.lm)

    collator = BatchCollator(ZipCollator(
        [TextCollator(tokenizer), GraphCollator]))

    def dataload(dataset):
        return DataLoader(dataset, batch_size=ARGS.batch_size, shuffle=True, collate_fn=collator)

    train_dataloader = dataload(train_dataset)
    test_dataloader = dataload(test_dataset)

    language_model = AutoModel.from_pretrained(ARGS.lm)
    lm_dim = language_model.config.hidden_size

    gnn = load_gnn(ARGS.gnn_name, lm_dim, embed_dim, num_rels)
    model = LM_GNN(language_model, gnn)
    optimizer = AdamW(model.parameters(), lr=ARGS.lr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    wandb.init(
        project=ARGS.wandb_project,
        config=dict(
            kb_name=ARGS.kb_path,
            rel_path=ARGS.rel_path,
            emb_name=ARGS.node_embeddings,
            qa_task=ARGS.train_file,
            lm_name=ARGS.lm,
            gnn_name=ARGS.gnn_name,
            batch_size=ARGS.batch_size,
            epochs=ARGS.epochs,
            learning_rate=ARGS.lr,
            top_k=ARGS.top_k,
            num_hops=ARGS.hops,
        ),
    )

    best_acc = -1
    no_improvement = 0

    for epoch in range(ARGS.epochs):
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
            accuracy = correct / len(test_dataloader.dataset)
            if accuracy > best_acc:
                best_acc = accuracy
                no_improvement = 0
            else:
                no_improvement += 1
            wandb.log({"validation/accuracy": accuracy})
            if no_improvement > ARGS.max_no_improvement:
                break


if __name__ == "__main__":
    train()
