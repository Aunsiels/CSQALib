import csv
import json
from tqdm import tqdm

import pandas as pd

from .common import DF_EDGES_COLS


def load_cpnet(path: str):
    """
    - reads the csv file into a pd.DataFrame
    - dropna
    """
    rows = []
    with open(path) as fin:
        for _, rel, head, tail, info in tqdm(
            csv.reader(fin, delimiter="\t"), desc="read_cpnet"
        ):
            info = json.loads(info)
            rows.append(
                (
                    rel.split("/", 2)[2].lower(),
                    head.split("/")[3],
                    tail.split("/")[3],
                    info.get("weight", 0.0),
                )
            )
    edges = pd.DataFrame(rows, columns=DF_EDGES_COLS).dropna()
    return edges
