import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import gensim.downloader

from .lm_utils import mean_pooling


class Embedder:
    @property
    def dim(self) -> int:
        return self._dim

    def __call__(self, cids: list) -> np.array:
        raise NotImplementedError


class TransE_Embedder(Embedder):
    def __init__(self, embeddings: np.array, cids):
        self.cid2idx = {cid: i for i, cid in enumerate(cids)}
        self.embeddings = embeddings
        self._dim = embeddings.shape[1]

    def __call__(self, cids) -> np.array:
        indices = [self.cid2idx[cid] for cid in cids]
        return self.embeddings[indices]


class GloVe_Embedder(Embedder):
    def __init__(self, model='glove-wiki-gigaword-300'):
        # will be cached after first use
        self.wv = gensim.downloader.load(model)

        some_vec = self.wv['man']
        oov_vec = np.random.randn(*some_vec.shape)
        self.oov = oov_vec * np.linalg.norm(some_vec) / np.linalg.norm(oov_vec)
        self._dim = some_vec.shape[0]

    def embed_one(self, cid):
        return np.mean([
            self.wv[w]
            if w in self.wv else
            self.oov
            for w in cid.lower().split('_')
        ], axis=0)

    def __call__(self, cids) -> np.array:
        if len(cids) == 0:
            return np.zeros((0, self.dim))
        return np.stack([self.embed_one(cid) for cid in cids])


class LM_Embedder(Embedder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self._dim = self.model.config.hidden_size

    def __call__(self, cids) -> torch.Tensor:
        concepts = [cid.replace("_", " ") for cid in cids]
        encoded_input = self.tokenizer(
            concepts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
        embeddings = self.model(**encoded_input)
        return mean_pooling(embeddings, encoded_input['attention_mask']).to('cpu')
