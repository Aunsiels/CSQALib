import pandas as pd

from loaders.cpnet import load_cpnet
from loaders.csqa import load_csqa
from loaders.spor_kb import load_spor_kb
from utils.concept_embedder import Embedder, GloVe_Embedder, LM_Embedder

LOADERS = {
    "spor": load_spor_kb,
    "conceptnet": load_cpnet
}


def load_edges(kb_type, kb_path):
    return LOADERS[kb_type](kb_path)


def load_relations(rel_path):
    return pd.read_csv(rel_path, sep='\t')


def load_embedder(emb_name) -> Embedder:
    if emb_name == "glove":
        return GloVe_Embedder()
    if "sentence-transformers" in emb_name:
        return LM_Embedder(emb_name)
    # TODO TransE
    raise NotImplementedError


def load_qa(train_file, test_file):
    train = load_csqa(train_file)
    test = load_csqa(test_file)
    return train, test