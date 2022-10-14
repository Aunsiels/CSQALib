from loaders.loader_factory import load_edges, load_relations
from parameters import ARGS
from utils.build_graph import build_graph
from utils.sample_paths import sample_paths
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset


def path_to_sent(path):
    # (first, rel, ..., rel, last) -> last <SEP> first rel .. rel last
    words = [x.replace("_", " ") if i %
             2 == 0 else x for i, x in enumerate(path)]
    return words[-1] + "[SEP]" + ' '.join(words)


def train_embed(export_path="data/path-gen.pt"):
    edges = load_edges(ARGS.kb_type, ARGS.kb_path)
    relations = load_relations(ARGS.rel_path)
    graph, _ = build_graph(edges, relations, use_idx=True)
    graph.remove_nodes_from([n for n in graph.nodes if not graph.degree(n)])

    # prep
    paths = sample_paths(graph)
    text_dataset = [path_to_sent(path) for path in paths]
    dataset = HFDataset.from_dict({"text": text_dataset})
    dataset = dataset.map((lambda batch: tokenizer(
        batch)), batched=True, remove_columns=['text'])
    dataloader = DataLoader(dataset, batch_size=ARGS.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    optimizer = AdamW(model.parameters(), lr=ARGS.lr)

    for epoch in range(ARGS.epochs):
        for batch in dataloader:
            output = model(**batch.to(device))
            loss = output[0]

            model.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), export_path)


def train_qa(model_path="data/path-gen.pt"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(torch.load(model_path))
    