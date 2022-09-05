# Setup

```sh
# download data
(mkdir -p data && cd data && make -f ../Makefile)

pip install torch
TORCH=$(python -c 'import torch; print(torch.__version__)')
pip install torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-${TORCH}.html"
pip install git+https://github.com/pyg-team/pytorch_geometric.git
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c 'import nltk; nltk.download("stopwords")'
wandb login  # or wandb disabled
```

# Run

```sh
python train_qagnn.py optional/config.json
```