"""
wget -O quasimodo_positive_top.tsv https://nextcloud.mpi-klsb.mpg.de/index.php/s/Sioq6rKP8LmjMDQ/download?path=%2FLatest%2fquasimodo_positive_top.tsv
"""
import pandas as pd

from .common import DF_EDGES_COLS


QUASIMODO_COLUMNS = [
    'subject',
    'predicate',
    'object',
    'modality',
    'is_negative',
    'score',
    'stuff',
    'neighborhood sigma',
    'local sigma',
]


def load_quasimodo(path):
    return (
        pd.read_csv(path, sep='\t', names=QUASIMODO_COLUMNS,
                    encoding='latin-1')
        .rename({"subject": "Head", "predicate": "Type", "object": "Tail", "score": "Weight"})
        [DF_EDGES_COLS]
    )
