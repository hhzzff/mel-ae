from typing import Dict, Optional

import torch
import torchaudio
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
from audiotools.ml import BaseModel
from vocos import Vocos

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s'
)

def init_module_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block.
    DwConv -> Permute to (N, T, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    
    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, T) -> (N, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1) # (N, T, C) -> (N, C, T)

        x = input_x + x
        return x

class ConvNeXtEncoder(nn.Module):
    r""" ConvNeXt Encoder
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_channels (int): Number of input image channels. Default: 100
        depth (int): Number of ConvNeXtBlocks. Default: 8
        dim (int): Feature dimension. Default: 512
        out_channels (int): Number of output image channels. Default: 128
    """
    def __init__(self, in_channels=100, depth=8, hidden_dim=512, out_channels=128, stride=4):
        super().__init__()

        self.downsample_layer = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=stride * 2, stride=stride, padding=math.ceil(stride / 2),),
            # Permute(0, 2, 1),
            # nn.LayerNorm(hidden_dim, eps=1e-6),
            # Permute(0, 2, 1),
            # nn.PReLU(num_parameters=hidden_dim),
        )

        self.stage = nn.Sequential(
            *[ConvNeXtBlock(dim=hidden_dim) for _ in range(depth)]
        )

        self.output_proj = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.downsample_layer(x)
        logging.debug(f"x.shape:{x.shape}")
        x = self.stage(x)
        logging.debug(f"x.shape:{x.shape}")
        x = self.output_proj(x)
        return x


class ConvNeXtDecoder(nn.Module):
    r""" ConvNeXt Decoder
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_channels (int): Number of input image channels. Default: 128
        depth (int): Number of ConvNeXtBlocks. Default: 8
        dim (int): Feature dimension. Default: 512
        out_channels (int): Number of output image channels. Default: 100
    """
    def __init__(self, in_channels=128, depth=8, hidden_dim=512, out_channels=100, stride=4):
        super().__init__()
        
        self.input_proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim, 
            kernel_size=1,
        )

        self.stage = nn.Sequential(
            *[ConvNeXtBlock(dim=hidden_dim) for _ in range(depth)]
        )

        self.upsample_layer = nn.Sequential(
            # Permute(0, 2, 1),
            # nn.LayerNorm(hidden_dim, eps=1e-6),
            # Permute(0, 2, 1),
            # nn.PReLU(num_parameters=hidden_dim),
            nn.ConvTranspose1d(hidden_dim, out_channels, kernel_size=stride * 2, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        x = self.input_proj(x)
        logging.debug(f"x.shape:{x.shape}")
        x = self.stage(x)
        logging.debug(f"x.shape:{x.shape}")
        x = self.upsample_layer(x)
        return x
    
class MelVAE(nn.Module):
    """ConvNeXt-based mel spectrogram autoencoder."""

    def __init__(
        self,
        in_channels: int = 100,
        out_channels: int = 100,
        latent_dim: int = 128,
        depth: int = 8,
        hidden_dim: int = 512,
        stride: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = ConvNeXtEncoder(
            in_channels=in_channels,
            depth=depth,
            hidden_dim=hidden_dim,
            out_channels=latent_dim,
            stride=stride,
        )
        self.logvar_proj = nn.Conv1d(latent_dim, latent_dim, 1)
        self.decoder = ConvNeXtDecoder(
            in_channels=latent_dim,
            depth=depth,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            stride=stride,
        )
        self.apply(init_module_weights)
    
    def noise(self, z):
        if self.training:
            logvar = self.logvar_proj(z)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return z + eps * std, logvar
        else:
            return z, None

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        original_length = x.size(-1)
        pad_len = (4 - original_length % 4) % 4
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        z = self.encoder(x)
        noise_z, logvar = self.noise(z)
        rec = self.decoder(noise_z)

        return {
            "rec": rec[..., :original_length],
            "z": z,
            "noise_z": noise_z,
            "logvar": logvar,
        }

class MetaMelVAE(BaseModel):
    """ConvNeXt-based mel spectrogram autoencoder."""
    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 8,
        hidden_dim: int = 512,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        stride: int = 4,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )
        self.MelVAE = MelVAE(
            in_channels=n_mels,
            out_channels=n_mels,
            latent_dim=latent_dim,
            depth=depth,
            hidden_dim=hidden_dim,
            stride=stride,
        )
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.vocoder.eval()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def extract_feature(self, audio, **kwargs):
        mel = self.mel_spec(audio)
        features = torch.log(torch.clip(mel, min=1e-7))
        return features

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: Optional[int] = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
            "mel_in" : Tensor[B x n_mels x length]
                Input logmel.
            "mel_out" : Tensor[B x n_mels x length]
                Decoded logmel.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        mel = self.extract_feature(audio_data, sample_rate=sample_rate).squeeze()
        logging.debug(f"mel.shape:{mel.shape}")
        outputs = self.MelVAE(mel)
        with torch.no_grad():
            audio_recon = self.vocoder.decode(outputs["rec"])
        return {
            "audio": audio_recon[..., :length],
            "mel_in": mel,
            "mel_out": outputs["rec"][..., :mel.size(-1)],
            "z": outputs["z"],
            "noise_z": outputs["noise_z"],
            "logvar": outputs["logvar"],
        }

if __name__ == "__main__":
    from torch.nn.utils.parametrize import is_parametrized

    model = MetaMelVAE()

    for name, module in model.named_modules():
        if is_parametrized(module):
            print(f"Module {name} is parametrized!")