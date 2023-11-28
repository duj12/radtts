import numpy as np
import os
import yaml
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import kaiser
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm, spectral_norm
from distutils.version import LooseVersion
from pytorch_wavelets import DWT1DForward
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False
        )
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Design prototype filter for PQMF.

    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.

    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.

    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).

    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427

    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    """PQMF module.

    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122

    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        """Initilize PQMF module.

        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.

        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.

        """
        super(PQMF, self).__init__()

        # build analysis & synthesis filter coefficients
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    - (-1) ** k * np.pi / 4
                )
            )

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.

        Args:
            x (Tensor): Input tensor (B, 1, T).

        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).

        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.

        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).

        Returns:
            Tensor: Output tensor (B, 1, T).

        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = F.conv_transpose1d(
            x, self.updown_filter * self.subbands, stride=self.subbands
        )
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)


class Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv1d, self).__init__()
        self.conv1d = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
        )
        self.conv1d.apply(init_weights)

    def forward(self, x):
        x = self.conv1d(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1d)


class CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1d = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
        )
        self.conv1d.apply(init_weights)

    def forward(self, x):  # bdt
        x = F.pad(
            x, (self.pad, 0, 0, 0, 0, 0), "constant"
        )  # described starting from the last dimension and moving forward.
        #  x = F.pad(x, (self.pad, self.pad, 0, 0, 0, 0), "constant")
        x = self.conv1d(x)[:, :, : x.size(2)]
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1d)


class ConvTranspose1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
    ):
        super(ConvTranspose1d, self).__init__()
        self.deconv = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                output_padding=0,
            )
        )
        self.deconv.apply(init_weights)

    def forward(self, x):
        return self.deconv(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.deconv)


#  FIXME: HACK to get shape right
class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
    ):
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                output_padding=0,
            )
        )
        self.stride = stride
        self.deconv.apply(init_weights)
        self.pad = kernel_size - stride

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).
        Returns:
            Tensor: Output tensor (B, out_channels, T_out).
        """
        #  x = F.pad(x, (self.pad, 0, 0, 0, 0, 0), "constant")
        return self.deconv(x)[:, :, : -self.pad]
        #  return self.deconv(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.deconv)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        causal=False,
    ):
        super(ResidualBlock, self).__init__()
        assert kernel_size % 2 == 1, "Kernal size must be odd number."
        conv_cls = CausalConv1d if causal else Conv1d
        self.convs1 = nn.ModuleList(
            [
                conv_cls(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[i],
                    padding=get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                conv_cls(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
                for i in range(len(dilation))
            ]
        )

        self.activation = getattr(torch.nn, nonlinear_activation)(
            **nonlinear_activation_params
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.activation(x)
            xt = c1(xt)
            xt = self.activation(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            layer.remove_weight_norm()
        for layer in self.convs2:
            layer.remove_weight_norm()


class SourceModule(torch.nn.Module):
    def __init__(
        self, nb_harmonics, upsample_ratio, sampling_rate, alpha=0.1, sigma=0.003
    ):
        super(SourceModule, self).__init__()

        self.nb_harmonics = nb_harmonics
        self.upsample_ratio = upsample_ratio
        self.sampling_rate = sampling_rate
        self.alpha = alpha
        self.sigma = sigma

        self.ffn = nn.Sequential(
            weight_norm(nn.Conv1d(self.nb_harmonics + 1, 1, kernel_size=1, stride=1)),
            nn.Tanh(),
        )

    def forward(self, pitch, uv):
        """
        :param pitch: [B, 1, frame_len], Hz
        :param uv: [B, 1, frame_len] vuv flag
        :return: [B, 1, sample_len]
        """
        with torch.no_grad():
            pitch_samples = F.interpolate(
                pitch, scale_factor=(self.upsample_ratio), mode="nearest"
            )
            uv_samples = F.interpolate(
                uv, scale_factor=(self.upsample_ratio), mode="nearest"
            )

            F_mat = torch.zeros(
                (pitch_samples.size(0), self.nb_harmonics + 1, pitch_samples.size(-1))
            ).to(pitch_samples.device)
            for i in range(self.nb_harmonics + 1):
                F_mat[:, i : i + 1, :] = pitch_samples * (i + 1) / self.sampling_rate

            theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
            u_dist = Uniform(low=-np.pi, high=np.pi)
            phase_vec = u_dist.sample(
                sample_shape=(pitch.size(0), self.nb_harmonics + 1, 1)
            ).to(F_mat.device)
            phase_vec[:, 0, :] = 0

            n_dist = Normal(loc=0.0, scale=self.sigma)
            noise = n_dist.sample(
                sample_shape=(
                    pitch_samples.size(0),
                    self.nb_harmonics + 1,
                    pitch_samples.size(-1),
                )
            ).to(F_mat.device)

            e_voice = self.alpha * torch.sin(theta_mat + phase_vec) + noise
            e_unvoice = self.alpha / 3 / self.sigma * noise

            e = e_voice * uv_samples + e_unvoice * (1 - uv_samples)

        return self.ffn(e)

    def remove_weight_norm(self):
        remove_weight_norm(self.ffn[0])


class Generator(torch.nn.Module):
    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernal_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        repeat_upsample=True,
        bias=True,
        causal=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        nsf_params=None,
    ):
        super(Generator, self).__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernal size must be odd number."
        assert len(upsample_scales) == len(upsample_kernal_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        self.upsample_scales = upsample_scales
        self.repeat_upsample = repeat_upsample
        self.num_upsamples = len(upsample_kernal_sizes)
        self.num_kernels = len(resblock_kernel_sizes)
        self.out_channels = out_channels
        self.nsf_enable = nsf_params is not None

        self.transpose_upsamples = torch.nn.ModuleList()
        self.repeat_upsamples = torch.nn.ModuleList()  # for repeat upsampling
        self.conv_blocks = torch.nn.ModuleList()

        conv_cls = CausalConv1d if causal else Conv1d
        conv_transposed_cls = CausalConvTranspose1d if causal else ConvTranspose1d

        self.conv_pre = conv_cls(
            in_channels, channels, kernel_size, 1, padding=(kernel_size - 1) // 2
        )

        for i in range(len(upsample_kernal_sizes)):
            self.transpose_upsamples.append(
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    conv_transposed_cls(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_kernal_sizes[i],
                        upsample_scales[i],
                        padding=(upsample_kernal_sizes[i] - upsample_scales[i]) // 2,
                    ),
                )
            )

            if repeat_upsample:
                self.repeat_upsamples.append(
                    nn.Sequential(
                        nn.Upsample(mode="nearest", scale_factor=upsample_scales[i]),
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        conv_cls(
                            channels // (2 ** i),
                            channels // (2 ** (i + 1)),
                            kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                )

            for j in range(len(resblock_kernel_sizes)):
                self.conv_blocks.append(
                    ResidualBlock(
                        channels=channels // (2 ** (i + 1)),
                        kernel_size=resblock_kernel_sizes[j],
                        dilation=resblock_dilations[j],
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        causal=causal,
                    )
                )

        self.conv_post = conv_cls(
            channels // (2 ** (i + 1)),
            out_channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )

        if self.nsf_enable:
            self.source_module = SourceModule(
                nb_harmonics=nsf_params["nb_harmonics"],
                upsample_ratio=np.cumprod(self.upsample_scales)[-1],
                sampling_rate=nsf_params["sampling_rate"],
            )
            self.source_downs = nn.ModuleList()
            self.downsample_rates = [1] + self.upsample_scales[::-1][:-1]
            self.downsample_cum_rates = np.cumprod(self.downsample_rates)

            for i, u in enumerate(self.downsample_cum_rates[::-1]):
                if u == 1:
                    self.source_downs.append(
                        Conv1d(1, channels // (2 ** (i + 1)), 1, 1)
                    )
                else:
                    self.source_downs.append(
                        conv_cls(
                            1,
                            channels // (2 ** (i + 1)),
                            u * 2,
                            u,
                            padding=u // 2,
                        )
                    )

    def forward(self, x):
        if self.nsf_enable:
            mel = x[:, :-2, :]
            pitch = x[:, -2:-1, :]
            uv = x[:, -1:, :]
            excitation = self.source_module(pitch, uv)
        else:
            mel = x

        x = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            #  FIXME: sin function here seems to be causing issues
            x = torch.sin(x) + x
            rep = self.repeat_upsamples[i](x)
            # transconv
            up = self.transpose_upsamples[i](x)

            if self.nsf_enable:
                # Downsampling the excitation signal
                e = self.source_downs[i](excitation)
                # augment inputs with the excitation
                x = rep + e + up[:, :, : rep.shape[-1]]
            else:
                x = rep + up[:, :, : rep.shape[-1]]

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.conv_blocks[i * self.num_kernels + j](x)
                else:
                    xs += self.conv_blocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.transpose_upsamples:
            layer[-1].remove_weight_norm()
        for layer in self.repeat_upsamples:
            layer[-1].remove_weight_norm()
        for layer in self.conv_blocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()
        if self.nsf_enable:
            self.source_module.remove_weight_norm()
            for layer in self.source_downs:
                layer.remove_weight_norm()


class PeriodDiscriminator(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=[5, 3],
        channels=32,
        downsample_scales=[3, 3, 3, 3, 1],
        max_downsample_channels=1024,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_spectral_norm=False,
    ):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        in_chs, out_chs = in_channels, channels

        for downsample_scale in downsample_scales:
            self.convs.append(
                torch.nn.Sequential(
                    norm_f(
                        nn.Conv2d(
                            in_chs,
                            out_chs,
                            (kernel_sizes[0], 1),
                            (downsample_scale, 1),
                            padding=((kernel_sizes[0] - 1) // 2, 0),
                        )
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            )
            in_chs = out_chs
            out_chs = min(out_chs * 4, max_downsample_channels)

        self.conv_post = nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(
        self,
        periods=[2, 3, 5, 7, 11],
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_spectral_norm": False,
        },
    ):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [PeriodDiscriminator(**params)]

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs


class ScaleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=[15, 41, 5, 3],
        channels=128,
        max_downsample_channels=1024,
        max_groups=16,
        bias=True,
        downsample_scales=[2, 2, 4, 4, 1],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_spectral_norm=False,
    ):
        super(ScaleDiscriminator, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        self.convs = nn.ModuleList()

        self.convs.append(
            torch.nn.Sequential(
                norm_f(
                    nn.Conv1d(
                        in_channels,
                        channels,
                        kernel_sizes[0],
                        bias=bias,
                        padding=(kernel_sizes[0] - 1) // 2,
                    )
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        )
        in_chs = channels
        out_chs = channels
        groups = 4

        for downsample_scale in downsample_scales:
            self.convs.append(
                torch.nn.Sequential(
                    norm_f(
                        nn.Conv1d(
                            in_chs,
                            out_chs,
                            kernel_size=kernel_sizes[1],
                            stride=downsample_scale,
                            padding=(kernel_sizes[1] - 1) // 2,
                            groups=groups,
                            bias=bias,
                        )
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            )
            in_chs = out_chs
            out_chs = min(in_chs * 2, max_downsample_channels)
            groups = min(groups * 4, max_groups)

        out_chs = min(in_chs * 2, max_downsample_channels)
        self.convs.append(
            torch.nn.Sequential(
                norm_f(
                    nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[2],
                        stride=1,
                        padding=(kernel_sizes[2] - 1) // 2,
                        bias=bias,
                    )
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        )

        self.conv_post = norm_f(
            nn.Conv1d(
                out_chs,
                out_channels,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=(kernel_sizes[3] - 1) // 2,
                bias=bias,
            )
        )

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        scales=3,
        downsample_pooling="DWT",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=False,
    ):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                params["use_spectral_norm"] = True if i == 0 else False
            self.discriminators += [ScaleDiscriminator(**params)]

        if downsample_pooling == "DWT":
            self.meanpools = nn.ModuleList(
                [DWT1DForward(wave="db3", J=1), DWT1DForward(wave="db3", J=1)]
            )
            self.aux_convs = nn.ModuleList(
                [
                    weight_norm(nn.Conv1d(2, 1, 15, 1, padding=7)),
                    weight_norm(nn.Conv1d(2, 1, 15, 1, padding=7)),
                ]
            )
        else:
            self.meanpools = nn.ModuleList(
                [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
            )
            self.aux_convs = None

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                if self.aux_convs is None:
                    y = self.meanpools[i - 1](y)
                else:
                    yl, yh = self.meanpools[i - 1](y)
                    y = torch.cat([yl, yh[0]], dim=1)
                    y = self.aux_convs[i - 1](y)
                    y = F.leaky_relu(y, 0.1)

            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs


class SpecDiscriminator(torch.nn.Module):
    def __init__(
        self,
        channels=32,
        init_kernel=15,
        kernel_size=11,
        stride=2,
        use_spectral_norm=False,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        super(SpecDiscriminator, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        # fft_size // 2 + 1
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        final_kernel = 5
        post_conv_kernel = 3
        blocks = 3  # TODO: remove hard code here
        self.convs = nn.ModuleList()
        self.convs.append(
            torch.nn.Sequential(
                norm_f(
                    nn.Conv2d(
                        fft_size // 2 + 1,
                        channels,
                        (init_kernel, 1),
                        (1, 1),
                        padding=(init_kernel - 1) // 2,
                    )
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        )

        for i in range(blocks):
            self.convs.append(
                torch.nn.Sequential(
                    norm_f(
                        nn.Conv2d(
                            channels,
                            channels,
                            (kernel_size, 1),
                            (stride, 1),
                            padding=(kernel_size - 1) // 2,
                        )
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            )

        self.convs.append(
            torch.nn.Sequential(
                norm_f(
                    nn.Conv2d(
                        channels,
                        channels,
                        (final_kernel, 1),
                        (1, 1),
                        padding=(final_kernel - 1) // 2,
                    )
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        )

        self.conv_post = norm_f(
            nn.Conv2d(
                channels,
                1,
                (post_conv_kernel, 1),
                (1, 1),
                padding=((post_conv_kernel - 1) // 2, 0),
            )
        )
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, wav):
        with torch.no_grad():
            wav = torch.squeeze(wav, 1)
            x_mag = stft(
                wav, self.fft_size, self.shift_size, self.win_length, self.window
            )
            x = torch.transpose(x_mag, 2, 1).unsqueeze(-1)
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.squeeze(-1)

        return x, fmap


class MultiSpecDiscriminator(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        discriminator_params={
            "channels": 15,
            "init_kernel": 1,
            "kernel_sizes": 11,
            "stride": 2,
            "use_spectral_norm": False,
            "window": "hann_window",
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
    ):
        super(MultiSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
            params = copy.deepcopy(discriminator_params)
            params["fft_size"] = fft_size
            params["shift_size"] = hop_size
            params["win_length"] = win_length
            self.discriminators += [SpecDiscriminator(**params)]

    def forward(self, y):
        y_d = []
        fmap = []
        for i, d in enumerate(self.discriminators):
            x, x_map = d(y)
            y_d.append(x)
            fmap.append(x_map)

        return y_d, fmap


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()

        self.audio2mel = Audio2Mel(filter_length,
                                   hop_length,
                                   win_length,
                                   sampling_rate,
                                   n_mel_channels,
                                   mel_fmin,
                                   mel_fmax)
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, y):
        """ see audio2mel
                """
        assert (torch.min(y.data) >= -1), torch.min(y.data)
        assert (torch.max(y.data) <= 1)

        mel_output = self.audio2mel.mel_spectrogram(y)
        return mel_output

    def mel_spectrogram(self, y):
        """ see audio2mel
        """
        assert (torch.min(y.data) >= -1), torch.min(y.data)
        assert (torch.max(y.data) <= 1)

        mel_output = self.audio2mel.mel_spectrogram(y)
        return mel_output


class Audio2Mel(torch.nn.Module):
    """
    This class uses Torch STFT, introduced by the MelGAN model
    """
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ):
        super(Audio2Mel, self).__init__()
        ##############################################
        # FFT Parameters
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.min_mel = math.log(1e-5)

    def forward(self, audio, normalize=False):
        """
        INPUT audio: B x 1 x T, range(-1, 1)
        OUTPUT log mel: B x D x T', T' is a reduction of T
        """
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        real_part, imag_part = torch.view_as_real(fft).unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(
            mel_output, min=math.exp(self.min_mel)))

        if normalize:
            log_mel_spec = (log_mel_spec - self.min_mel) / -self.min_mel

        return log_mel_spec

    def mel_spectrogram(self, audio_norm):
        audio_norm = audio_norm.unsqueeze(1)
        return self.forward(audio_norm)


def load_model(ckpt, config=None):
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(os.path.dirname(ckpt))
        config = os.path.join(dirname, "config.yaml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error

    model = Generator(**config["Model"]["Generator"]["params"])
    states = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(states["model"]["generator"])

    # add pqmf if needed
    if config["Model"]["Generator"]["params"]["out_channels"] > 1:
        # lazy load for circular error
        model.pqmf = PQMF()

    return model


def binarize(mel, threshold=0.6):
    # vuv binarize
    res_mel = mel.copy()
    index = np.where(mel[:, -1] < threshold)[0]
    res_mel[:, -1] = 1.0
    res_mel[:, -1][index] = 0.0
    return res_mel


def hifigan_infer(input_mel, ckpt_path, output_dir, config=None):
    import logging, time, glob, librosa
    import soundfile as sf
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda", 0)

    if config is not None:
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml"
        )
        if not os.path.exists(config_path):
            raise ValueError("config file not found: {}".format(config_path))
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # check directory existence
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isfile(input_mel):
        mel_lst = [input_mel]
    elif os.path.isdir(input_mel):
        mel_lst = glob.glob(os.path.join(input_mel, "*.npy"))
    else:
        raise ValueError("input_mel should be a file or a directory")

    model = load_model(ckpt_path, config)

    logging.info(f"Loaded model parameters from {ckpt_path}.")
    model.remove_weight_norm()
    model = model.eval().to(device)

    with torch.no_grad():
        start = time.time()
        pcm_len = 0
        for mel in mel_lst:
            utt_id = os.path.splitext(os.path.basename(mel))[0]
            if mel.endswith('.npy'):
                mel_data_npy = np.load(mel)  # T, C
            elif input_mel.endswith(".wav"):
                # wave = torch.from_numpy(
                #     librosa.load(input_mel, sr=16000, mono=True)[0])
                # mel_generator = TacotronSTFT(
                #     hop_length=200, sampling_rate=16000,
                #     filter_length=1024, win_length=1024
                # )
                # mel_data = mel_generator(wave.unsqueeze(0))   # B C T
                # mel_data_npy = mel_data.squeeze(0).transpose(0,1).cpu().numpy()  # T C
                wave = librosa.load(input_mel, sr=16000, mono=True)[0]
                from dsp import melspectrogram
                mel_data_npy = melspectrogram(
                    wave, 16000, n_fft=2048, hop_length=200, fmin=0)

            if model.nsf_enable:
                mel_data_npy = binarize(mel_data_npy)  # T C
            mel_data = torch.from_numpy(
                mel_data_npy).transpose(0, 1).unsqueeze(0).to(device)  # B C T

            # generate
            y = model(mel_data)
            if hasattr(model, "pqmf"):
                y = model.pqmf.synthesis(y)
            y = y.view(-1).cpu().numpy()
            pcm_len += len(y)

            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(output_dir, f"{utt_id}_gen.wav"),
                y,
                config["audio_config"]["sampling_rate"],
                "PCM_16",
            )
        rtf = (time.time() - start) / (
            pcm_len / config["audio_config"]["sampling_rate"]
        )

    # report average RTF
    logging.info(
        f"Finished generation of {len(mel_lst)} utterances (RTF = {rtf:.03f})."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Infer hifigan model")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_mel",
        type=str,
        required=True,
        help="Path to input mel file or directory containing mel files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    hifigan_infer(
        args.input_mel,
        args.ckpt,
        args.output_dir,
        args.config,
    )