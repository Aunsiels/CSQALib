from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        seq = []
        for i in range(num_layers + 1):
            in_size = input_size if i == 0 else hidden_size
            out_size = hidden_size if i < num_layers else output_size
            seq.append(nn.Linear(in_size, out_size))
            if i < num_layers:
                seq.append(nn.ReLU())
                seq.append(nn.Dropout(dropout_rate))
        self.layers = nn.Sequential(*seq)
