# Loads a KB in the format subject\tpredicate\tobject\score
import pandas as pd


def load_spor_kb(path):
    df = pd.read_csv(path, sep='\t', names=["Head", "Type", "Tail", "Weight"],
                    encoding='latin-1', dtype={"Head": str, "Type": str, "Tail": str, "Weight": float}).dropna()
    df['Type'] = df['Type'].str.lower()
    return df
