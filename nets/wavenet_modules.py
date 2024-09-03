import torch
from torch import nn

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dilated_conv(x)
        tanh = self.tanh(x)
        sigmoid = self.sigmoid(x)
        return tanh * sigmoid


class WaveNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(WaveNetLayer, self).__init__()
        self.block = WaveNetBlock(in_channels, out_channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.block(x)
        residual = self.residual_conv(x)
        skip = self.skip_conv(x)
        return x + residual, skip


class WaveNetStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers):
        super(WaveNetStack, self).__init__()
        # dilation doubles at each layer, when reaches 512, it resets to 1
        dilations = []
        for i in range(num_layers):
            dilations.append(2 ** (i % 9))
        self.layers = nn.ModuleList([
            WaveNetLayer(in_channels, out_channels, kernel_size, dilation)
            for dilation in dilations
        ])

    def forward(self, x):
        skips = []
        for layer in self.layers:
            x, skip = layer(x)
            skips.append(skip)
        return sum(skips)


class WaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers):
        super(WaveNet, self).__init__()
        self.casual_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = self.casual_conv(x)[..., :-1]
        return x


if __name__ == '__main__':
    model = WaveNet(1, 1, 2, 20)
    # print(model)
    x = torch.randn(1, 1, 10)
    assert model(x).shape == torch.Size([1, 1, 10])