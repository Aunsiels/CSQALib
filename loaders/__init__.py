import pandas as pd

from .quasimodo import load_quasimodo
from .ascent import load_ascent
from .cpnet import load_cpnet
from .csqa import load_csqa
from .obqa import load_obqa
from .fake import load_fake_kb, load_fake_qa

from src.concept_embedder import Embedder, LM_Embedder, GloVe_Embedder


DATA_DIR = "data"


def load_edges(kb_name):
    return {
        "quasimodo": lambda: load_quasimodo(f"{DATA_DIR}/quasimodo_positive_top.tsv"),
        "ascent": lambda: load_ascent(f"{DATA_DIR}/ascentpp.csv"),
        "cpnet_en": lambda: load_cpnet(f"{DATA_DIR}/cpnet_en.csv"),
    }[kb_name]()


def load_rels(kb_name):
    return {
        "quasimodo": lambda: pd.read_csv(...),  # TODO
        "ascent": lambda: pd.read_csv(f"{DATA_DIR}/cpnet-rels.tsv", sep='\t'),
        "cpnet_en": lambda: pd.read_csv(f"{DATA_DIR}/cpnet-rels.tsv", sep='\t'),
    }[kb_name]()


def load_embedder(emb_name) -> Embedder:
    if emb_name == "glove":
        return GloVe_Embedder()
    if "sentence-transformers" in emb_name:
        return LM_Embedder(emb_name)
    # TODO TransE
    raise NotImplementedError


def load_qa(qa_task):
    if qa_task == "csqa":
        train = load_csqa(f"{DATA_DIR}/csqa/train_rand_split.jsonl")
        test = load_csqa(f"{DATA_DIR}/csqa/dev_rand_split.jsonl")
        return train, test
    if qa_task == "obqa":
        train = load_obqa(f"{DATA_DIR}/obqa/train.jsonl")
        test = load_obqa(f"{DATA_DIR}/obqa/dev.jsonl")
        return train, test
    raise NotImplementedError
