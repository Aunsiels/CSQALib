"""
wget https://www.mpi-inf.mpg.de/fileadmin/inf/d5/research/ascentpp/ascentpp.csv.tar.gz
tar -xvzf ascentpp.csv.tar.gz
"""

import pandas as pd

from .common import DF_EDGES_COLS

ASCENT_COLS = [
    'primary_subject',
    'subject_type',
    'head',
    'relation',
    'tail',
    'subject',
    'predicate',
    'object',
    'saliency',
    'typicality',
    'facets',
]


def load_ascent(path: str):
    return (
        pd.read_csv(path)
        .rename({"subect": "Head", "object": "Tail", "predicate": "Type"}, axis=1)
        .assign(Weight=1.)
        [DF_EDGES_COLS]
    )
