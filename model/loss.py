import torch
from torch import nn
import torch.nn.functional as F

class BatchRMSLoss(nn.Module):
    def __init__(self, rms_limit: float = 1.0):
        super().__init__()
        self.rms_limit = rms_limit

    def forward(self, x: torch.Tensor):
        batch_rms = (x ** 2).mean(dim=(1,2)).sqrt()
        loss = (batch_rms - self.rms_limit).clamp(min=0.0)
        return loss.mean(), batch_rms.detach().mean()

class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, x, y):
        loss = self.loss_fn(x, y)
        return loss

class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        loss_d += torch.mean(d_fake[-1] ** 2)
        loss_d += torch.mean((1 - d_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        loss_g += torch.mean((1 - d_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake) - 1):
            loss_feature += F.l1_loss(d_fake[i], d_real[i].detach())
        return loss_g, loss_feature