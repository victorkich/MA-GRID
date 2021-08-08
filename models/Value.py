import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Value(nn.Module):
    def __init__(self, dim_state, dim_hidden=128, activation=nn.LeakyReLU):
        super(Value, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden

        self.value = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, 1)
        )

        self.value.apply(init_weight)

    def forward(self, x):
        value = self.value(x)
        return value
