import torch


def make_one_hot(labels, classes):
    return torch.zeros((*labels.size(), classes)).scatter(dim=1, index=labels, src=1.)
