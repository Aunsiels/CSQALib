# Setup

```sh
# download data
(mkdir -p data && cd data && make -f ../Makefile)

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c 'import nltk; nltk.download("stopwords")'
wandb login
```

# Run

```sh
python train_qagnn.py optional/config.json
```