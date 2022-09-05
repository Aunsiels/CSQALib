import pandas as pd

def load_fake_qa():
    return [
        {"question": "What is the sun?", "choices": ["a car", "a star", "a house", "a planet", "a color"], "label": 1},
    ]

def load_fake_kb():
    return pd.DataFrame.from_records([
        {"Type": "isa", "Head": "sun", "Tail": "star", "Weight": 1.},
        {"Type": "atlocation", "Head": "sun", "Tail": "space", "Weight": 1.},
        {"Type": "atlocation", "Head": "planet", "Tail": "space", "Weight": 1.},
    ])