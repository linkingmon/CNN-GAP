# Feature Selection model

import torch
import torch.nn as nn


class FS(nn.Module):
    def __init__(self):
        super(FS, self).__init__()
        self.outlayer = nn.Linear(64, 3)

    def forward(self, x):
        x = self.outlayer(x)
        return x
