import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from audiotools.ml import BaseModel
def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))

class ResolutionDiscriminator(nn.Module):
    def __init__(
        self, channels=32
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                WNConv2d(1, channels, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1)),
                WNConv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1), act=False)
            ]
        )
    def forward(self, mel):
        x = mel.unsqueeze(1)
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        return fmap

class Discriminator(BaseModel):
    def __init__(
        self,
    ):
        super().__init__()
        self.discriminator = ResolutionDiscriminator()

    def forward(self, x):
        fmaps = self.discriminator(x)
        return fmaps
