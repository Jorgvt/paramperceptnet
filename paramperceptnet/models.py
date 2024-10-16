from typing import Any

import flax.linen as nn

from fxlayers.layers import *
from .layers import *


class PerceptNet(nn.Module):
    """IQA model inspired by the visual system."""

    config: Any

    @nn.compact
    def __call__(
        self,
        inputs,  # Assuming fs = 128 (cpd)
        **kwargs,
    ):
        ## (Independent) Color equilibration (Gamma correction)
        ## Might need to be the same for each number
        ## bias = 0.1 / kernel = 0.5
        if self.config.USE_GAMMA:
            outputs = GDNGamma()(inputs)
        else:
            outputs = GDN(kernel_size=(1, 1), apply_independently=True)(inputs)

        ## Color (ATD) Transformation
        outputs = nn.Conv(features=3, kernel_size=(1, 1), use_bias=False, name="Color")(
            outputs
        )
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        ## GDN Star A - T - D [Separated]
        outputs = GDN(kernel_size=(1, 1), apply_independently=True)(outputs)

        ## Center Surround (DoG)
        ## Initialized so that 3 are positives and 3 are negatives and no interaction between channels is present
        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.CS_KERNEL_SIZE, mode="symmetric"
        )
        outputs = CenterSurroundLogSigmaK(
            features=3,
            kernel_size=self.config.CS_KERNEL_SIZE,
            fs=21,
            use_bias=False,
            padding="VALID",
        )(outputs, **kwargs)
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        ## GDN per channel with mean substraction in T and D (Spatial Gaussian Kernel)
        ### fs = 32 / kernel_size = (11,11) -> 0.32 > 0.02 --> OK!
        ## TO-DO: - Spatial Gaussian Kernel (0.02 deg) -> fs = 64/2 & 0.02*64/2 = sigma (px) = 0.69
        outputs = GDNGaussian(
            kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE,
            apply_independently=True,
            fs=32,
            padding="symmetric",
            normalize_prob=self.config.NORMALIZE_PROB,
            normalize_energy=self.config.NORMALIZE_ENERGY,
        )(outputs, **kwargs)

        ## GaborLayer per channel with GDN mixing only same-origin-channel information
        ### [Gaussian] sigma = 0.2 (deg) fs = 32 / kernel_size = (21,21) -> 21/32 = 0.66 --> OK!
        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.GABOR_KERNEL_SIZE, mode="symmetric"
        )
        # outputs, fmean, theta_mean = GaborLayerGamma_(n_scales=4+2+2, n_orientations=8*3, kernel_size=self.config.GABOR_KERNEL_SIZE, fs=32, xmean=self.config.GABOR_KERNEL_SIZE/32/2, ymean=self.config.GABOR_KERNEL_SIZE/32/2, strides=1, padding="VALID", normalize_prob=self.config.NORMALIZE_PROB, normalize_energy=self.config.NORMALIZE_ENERGY, zero_mean=self.config.ZERO_MEAN, use_bias=self.config.USE_BIAS, train_A=self.config.A_GABOR)(outputs, return_freq=True, return_theta=True, **kwargs)
        outputs, fmean, theta_mean = GaborLayerGammaHumanLike_(
            n_scales=[4, 2, 2],
            n_orientations=[8, 8, 8],
            kernel_size=self.config.GABOR_KERNEL_SIZE,
            fs=32,
            xmean=self.config.GABOR_KERNEL_SIZE / 32 / 2,
            ymean=self.config.GABOR_KERNEL_SIZE / 32 / 2,
            strides=1,
            padding="VALID",
            normalize_prob=self.config.NORMALIZE_PROB,
            normalize_energy=self.config.NORMALIZE_ENERGY,
            zero_mean=self.config.ZERO_MEAN,
            use_bias=self.config.USE_BIAS,
            train_A=self.config.A_GABOR,
        )(outputs, return_freq=True, return_theta=True, **kwargs)

        ## Final GDN mixing Gabor information (?)
        outputs = GDNSpatioChromaFreqOrient(
            # kernel_size=self.config.GDNFINAL_KERNEL_SIZE,
            kernel_size=21,
            strides=1,
            padding="symmetric",
            fs=32,
            apply_independently=False,
            normalize_prob=self.config.NORMALIZE_PROB,
            normalize_energy=self.config.NORMALIZE_ENERGY,
        )(outputs, fmean=fmean, theta_mean=theta_mean, **kwargs)

        return outputs


class Baseline(nn.Module):
    """IQA model inspired by the visual system."""

    config: Any

    @nn.compact
    def __call__(
        self,
        inputs,  # Assuming fs = 128 (cpd)
        **kwargs,
    ):
        outputs = GDN(kernel_size=(1, 1), apply_independently=True)(inputs)

        ## Color (ATD) Transformation
        outputs = nn.Conv(features=3, kernel_size=(1, 1), use_bias=False, name="Color")(
            outputs
        )
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        ## GDN Star A - T - D [Separated]
        outputs = GDN(kernel_size=(1, 1), apply_independently=True)(outputs)

        ## Center Surround (DoG)
        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.CS_KERNEL_SIZE, mode="symmetric"
        )
        outputs = nn.Conv(
            features=3,
            kernel_size=(self.config.CS_KERNEL_SIZE, self.config.CS_KERNEL_SIZE),
            use_bias=False,
            padding="VALID",
        )(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        ## GDN per channel with mean substraction in T and D (Spatial Gaussian Kernel)
        outputs = GDN(
            kernel_size=(
                self.config.GDNGAUSSIAN_KERNEL_SIZE,
                self.config.GDNGAUSSIAN_KERNEL_SIZE,
            ),
            apply_independently=True,
            padding="SAME",
        )(outputs)

        ## GaborLayer per channel with GDN mixing only same-origin-channel information
        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.GABOR_KERNEL_SIZE, mode="symmetric"
        )
        outputs = nn.Conv(
            features=128,
            kernel_size=(self.config.GABOR_KERNEL_SIZE, self.config.GABOR_KERNEL_SIZE),
            padding="VALID",
            use_bias=False,
        )(outputs)

        ## Final GDN mixing Gabor information (?)
        outputs = GDN(
            kernel_size=(
                self.config.GDNFINAL_KERNEL_SIZE,
                self.config.GDNFINAL_KERNEL_SIZE,
            ),
            apply_independently=False,
            padding="SAME",
        )(outputs)

        return outputs
