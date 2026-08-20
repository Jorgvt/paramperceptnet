from typing import Any
import flax.linen as nn

from fxlayers.layers import *
from .layers import *


class AblationPerceptNet(nn.Module):
    """Ablated IQA model that dynamically switches between parametric and baseline layers."""

    config: Any

    @nn.compact
    def __call__(
        self,
        inputs,  # Assuming fs = 128 (cpd)
        **kwargs,
    ):
        # Read dynamic config flags (default to True for full parametric behavior)
        use_gamma = getattr(self.config, "USE_GAMMA", True)
        use_param_cs = getattr(self.config, "PARAM_CS", True)
        use_param_dn_cs = getattr(self.config, "PARAM_DN_CS", True)
        use_param_gabor = getattr(self.config, "PARAM_GABOR", True)
        use_param_dn_final = getattr(self.config, "PARAM_DN_FINAL", True)
        use_final_b = getattr(self.config, "FINAL_B", True)

        # 1. Color equilibration (Gamma correction)
        if use_gamma:
            outputs = GDNGamma()(inputs)
        else:
            outputs = GDN(kernel_size=(1, 1), apply_independently=True)(inputs)

        # 2. Color (ATD) Transformation
        outputs = nn.Conv(features=3, kernel_size=(1, 1), use_bias=False, name="Color")(
            outputs
        )
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        # 3. GDN Star A - T - D [Separated]
        outputs = GDN(kernel_size=(1, 1), apply_independently=True)(outputs)

        # 4. Center Surround (DoG)
        outputs = pad_same_from_kernel_size(
            outputs, kernel_size=self.config.CS_KERNEL_SIZE, mode="symmetric"
        )
        if use_param_cs:
            outputs = CenterSurroundLogSigmaK(
                features=3,
                kernel_size=self.config.CS_KERNEL_SIZE,
                fs=21,
                use_bias=False,
                padding="VALID",
                normalize_sum=False,  # checkpoints predate this param; old default was False
            )(outputs, **kwargs)
        else:
            outputs = nn.Conv(
                features=3,
                kernel_size=(self.config.CS_KERNEL_SIZE, self.config.CS_KERNEL_SIZE),
                use_bias=False,
                padding="VALID",
            )(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2, 2), strides=(2, 2))

        # 5. GDN per channel (Spatial Divisive Normalization)
        if use_param_dn_cs:
            outputs = GDNGaussian(
                kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE,
                apply_independently=True,
                fs=32,
                padding="symmetric",
                normalize_prob=self.config.NORMALIZE_PROB,
                normalize_energy=self.config.NORMALIZE_ENERGY,
            )(outputs, **kwargs)
        else:
            outputs = GDN(
                kernel_size=(
                    self.config.GDNGAUSSIAN_KERNEL_SIZE,
                    self.config.GDNGAUSSIAN_KERNEL_SIZE,
                ),
                apply_independently=True,
                padding="SAME",
            )(outputs)

        # 6. GaborLayer per channel
        if use_param_gabor:
            outputs = pad_same_from_kernel_size(
                outputs, kernel_size=self.config.GABOR_KERNEL_SIZE, mode="symmetric"
            )
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
        else:
            outputs = pad_same_from_kernel_size(
                outputs, kernel_size=self.config.GABOR_KERNEL_SIZE, mode="symmetric"
            )
            outputs = nn.Conv(
                features=128,
                kernel_size=(self.config.GABOR_KERNEL_SIZE, self.config.GABOR_KERNEL_SIZE),
                padding="VALID",
                use_bias=False,
            )(outputs)
            fmean, theta_mean = None, None

        # 7. Final GDN mixing Gabor information
        if use_param_dn_final:
            outputs = GDNSpatioChromaFreqOrient(
                kernel_size=21,
                strides=1,
                padding="symmetric",
                fs=32,
                apply_independently=False,
                normalize_prob=self.config.NORMALIZE_PROB,
                normalize_energy=self.config.NORMALIZE_ENERGY,
            )(outputs, fmean=fmean, theta_mean=theta_mean, **kwargs)
        else:
            outputs = GDN(
                kernel_size=(
                    self.config.GDNFINAL_KERNEL_SIZE,
                    self.config.GDNFINAL_KERNEL_SIZE,
                ),
                apply_independently=False,
                padding="SAME",
            )(outputs)

        # 8. Final Scaling
        if use_final_b:
            outputs = self.param("B", nn.initializers.ones_init(), (outputs.shape[-1],)) * outputs

        return outputs
