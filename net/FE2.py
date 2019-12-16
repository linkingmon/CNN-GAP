# Feature Extraction model

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, dilation_rate, max_pooling):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=1, padding=dilation_rate//2*(kernel_size-1), dilation=dilation_rate)
        self.bn = nn.BatchNorm1d(out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = max_pooling
        self.maxpool = nn.MaxPool1d(2, 2, padding=0)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.max_pooling == True:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class FE(nn.Module):
    def __init__(self, final_len):
        super(FE, self).__init__()
        self.layer1 = ConvBlock(1, 256, 16, 2, True)
        self.layer2 = ConvBlock(256, 256, 16, 4, False)
        self.layer3 = ConvBlock(256, 256, 16, 4, False)
        self.layer4 = ConvBlock(256, 256, 16, 4, False)

        self.layer5 = ConvBlock(256, 128, 8, 4, True)
        self.layer6 = ConvBlock(128, 128, 8, 6, False)
        self.layer7 = ConvBlock(128, 128, 8, 6, False)
        self.layer8 = ConvBlock(128, 128, 8, 6, False)
        self.layer9 = ConvBlock(128, 128, 8, 6, False)

        self.layer10 = ConvBlock(128, 128, 8, 8, True)
        self.layer11 = ConvBlock(128, 64, 8, 8, False)
        self.layer12 = ConvBlock(64, 64, 8, 8, False)
        self.layer13 = nn.AvgPool1d(final_len)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        y = x
        x = self.layer13(x).reshape(x.size()[0], x.size()[1])
        return x, y
