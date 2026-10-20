"""
USDConvNetLite - a residual-connection design scaled down to a lightweight custom architecture.
"""

import torch
import torch.nn as nn

'''
Run input through two convolutions (and batch normalization layers), then add original unmodified input
back on top of result before final activation.
'''

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)

'''
Shrink the input a bit (stem).
Refine features through multiple residual blocks (4).
Bring down to fixed small size regardless of input shape (adapative pooling).
Output one score per class (linear layer).
Input:  (B, 1, T, C)  e.g. T=32 (freq+time stacked), C=15 (5 mags x xyz).
Output: (B, num_classes) logits.
num_classes = no. of output classes (slip/no-slip)
'''

class USDConvNetLite(nn.Module):

    def __init__(self, num_classes: int = 2, base_channels: int = 8, num_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.blocks = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(base_channels * 4 * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = USDConvNetLite(num_classes=2, base_channels=8, num_blocks=4)
    dummy = torch.randn(4, 1, 32, 15)
    out = m(dummy)
    print("output shape:", out.shape)
    print("parameter count:", count_parameters(m))