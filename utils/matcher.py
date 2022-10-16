"""
Matcher provides the interface
matcher(text) -> set of concept_ids

Requirements
python -m spacy download en_core_web_sm
python -c 'import nltk; nltk.download("stopwords")'
"""

from typing import List, Union
import spacy
from spacy.matcher import PhraseMatcher
import nltk
from tqdm import tqdm


NLTK_STOPWORDS = set(nltk.corpus.stopwords.words('english'))

MATCHER_BLACKLIST = {"-PRON-", "actually", "likely", "possibly", "want",
                     "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                     "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"}

PRONOUN_LIST = {"my", "you", "it", "its", "your", "i", "he",
                "she", "his", "her", "they", "them", "their", "our", "we"}


def is_pattern_ok(doc):
    return (
        len(doc) < 5 and
        doc[0].text not in PRONOUN_LIST and
        doc[-1].text not in PRONOUN_LIST and
        any(token.text not in NLTK_STOPWORDS and
            token.lemma_ not in NLTK_STOPWORDS and
            token.lemma_ not in MATCHER_BLACKLIST
            for token in doc)
    )


class Matcher:
    """
    Built from concept ids, where words are separated by '_'
    matcher.match_one(text) returns a set of concept ids.
    matcher.match_all([txt]) returns a list of sets of concept ids.
    """

    def __init__(self, concept_ids):
        self.nlp = nlp = spacy.load('en_core_web_sm', disable=[
                                    'ner', 'parser', 'textcat'])
        nlp.add_pipe('sentencizer')
        self.matcher = matcher = PhraseMatcher(nlp.vocab, "LEMMA")
        concept_spans = [cid.replace('_', ' ') for cid in concept_ids]

        for cid, doc in tqdm(zip(concept_ids, nlp.pipe(concept_spans)), desc="matcher", total=len(concept_ids)):
            if is_pattern_ok(doc):
                matcher.add(cid, [doc])

    def match_one(self, text: str) -> set:
        doc = self.nlp(text)
        return {self.nlp.vocab[cid].text for cid, _, _ in self.matcher(doc)}

    def match_all(self, corpus: List[str]) -> List[set]:
        return [{self.nlp.vocab[cid].text for cid, _, _ in self.matcher(doc)}
                for doc in self.nlp.pipe(corpus)]
