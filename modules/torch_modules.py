import torch
import torch.nn as nn


class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = (
            nn.Linear(in_dim if num_layers == 1 else inner_dim, out_dim)
            if num_layers >= 1
            else nn.Identity()
        )

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x
