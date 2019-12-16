# Feature Selection model

import torch
import torch.nn as nn


class FSwn(nn.Module):
    def __init__(self):
        super(FSwn, self).__init__()
        self.outlayer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.outlayer(x)
        return x
