import argparse

parser = argparse.ArgumentParser(description='Commonsense Question Answering')

parser.add_argument("--kb_path", default="data/empty.tsv", type=str, required=False,
                    help="The path to the knowledge base.")

parser.add_argument("--rel_path", default="data/empty_rel.tsv", type=str, required=False,
                    help="The relations used in the knowledge base and information about them.")

parser.add_argument("--kb_type", default="spor", type=str, required=False,
                    help="The type of knowledge base we are using, useful to know how to read it.")

parser.add_argument("--gnn_name", default="none", type=str, required=False,
                    help="The type of GNN to use.")

parser.add_argument("--train_file", type=str, required=True,
                    help="The training dataset.")

parser.add_argument("--test_file", type=str, required=True,
                    help="The testing dataset.")

parser.add_argument("--lm", default="roberta-large", type=str, required=False,
                    help="The language modeling used for the encoding. From transformers.")

parser.add_argument("--node_embeddings", default="glove", type=str, required=False,
                    help="The method used to compute embeddings for the nodes.")

parser.add_argument("--batch_size", default=16, type=int, required=False,
                    help="The batch size.")

parser.add_argument("--epochs", default=100, type=int, required=False,
                    help="The number of epochs.")

parser.add_argument("--lr", default=5e-5, type=float, required=False,
                    help="The learning rate.")

parser.add_argument("--top_k", default=100, type=int, required=False,
                    help="Limit of nodes to consider for each subgraph.")

parser.add_argument("--hops", default=3, type=int, required=False,
                    help="The number of hops.")

parser.add_argument("--wandb_project", default="csqa", type=str, required=False,
                    help="The name of the project in WANDB.")


ARGS = parser.parse_args()
