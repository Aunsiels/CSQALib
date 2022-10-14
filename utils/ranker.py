"""
ranker(context, [concept_ids]) -> [scores]
Don't forget to delete Ranker after use, to free up GPU mem
"""
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np


from .lm_utils import mean_pooling


class LMLossRanker:
    def __init__(self):
        model_name = "roberta-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name).to(device).eval()
        self.cache = {}

    def score_batch(self, pairs) -> List[float]:
        """
        paris = [(question, concept_id), ...]
        computes the score for this batch and adds new pairs to the cache.
        """
        as_text = [question + " " + cid.replace('_', ' ')
                   for (question, cid) in pairs]
        new_text = list(
            set(text for text in as_text if text not in self.cache))

        if new_text:
            tokenized = self.tokenizer(
                new_text, padding=True, return_tensors="pt")
            loss = self.model(**tokenized, labels=tokenized['input_ids']).loss
            for i, text in enumerate(new_text):
                self.cache[text] = loss[i, 0].item()

        return [self.cache[text] for text in as_text]


class CosineSimilarityRanker:
    def __init__(self):
        # sentence-transformers models are better suited for cosine similarity
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def __call__(self, context, concept_ids) -> np.array:
        # return an array of scores
        # shape: (len(concept_ids),)
        if len(concept_ids) == 0:
            return np.array([])
        concept_spans = [cid.replace('_', ' ') for cid in concept_ids]

        input = [context, *concept_spans]
        encoded_input = self.tokenizer(
            input, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask']).cpu()

        scores = cosine_similarity(
            sentence_embeddings[0:1], sentence_embeddings[1:])
        return scores[0]


def get_top_k(X, scores, k):
    ranking = np.argsort(-scores)
    return [X[i] for i in ranking[:k]]
