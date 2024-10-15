from typing import Sequence, Union, Callable

import jax
from jax import lax, numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat

from fxlayers.layers import *
from fxlayers.layers import GaussianLayerGamma
from fxlayers.initializers import *


class ChromaFreqOrientGaussianGamma(nn.Module):
    """(1D) Gaussian interaction between gamma_fuencies and orientations optimizing gamma = 1/sigma instead of sigma."""

    use_bias: bool = False
    strides: int = 1
    padding: str = "SAME"
    bias_init: Callable = nn.initializers.zeros_init()
    n_scales: Sequence[int] = jnp.array([4, 2, 2], dtype=jnp.int32)
    n_orientations: Sequence[int] = jnp.array([8, 8, 8], dtype=jnp.int32)

    @nn.compact
    def __call__(
        self,
        inputs,
        fmean,
        theta_mean,
        **kwargs,
    ):
        gamma_f_a = self.param(
            "gamma_f_a",
            k_array(1 / 0.4, arr=jnp.array([2.0, 4.0, 8.0, 16.0])),
            (self.n_scales[0],),
        )
        gamma_theta_a = self.param(
            "gamma_theta_a",
            nn.initializers.ones_init(),
            #  (self.n_orientations[0],))
            (8,),
        )

        gamma_f_t = self.param(
            "gamma_f_t",
            k_array(1 / 0.4, arr=jnp.array([3.0, 6.0])),
            (self.n_scales[1],),
        )
        gamma_theta_t = self.param(
            "gamma_theta_t",
            nn.initializers.ones_init(),
            #  (self.n_orientations[1],))
            (8,),
        )

        gamma_f_d = self.param(
            "gamma_f_d",
            k_array(1 / 0.4, arr=jnp.array([3.0, 6.0])),
            (self.n_scales[2],),
        )
        gamma_theta_d = self.param(
            "gamma_theta_d",
            nn.initializers.ones_init(),
            #  (self.n_orientations[2],))
            (8,),
        )

        H_cc = self.param("H_cc", nn.initializers.ones_init(), (3, 3))

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (len(fmean),))
        else:
            bias = 0.0
        # n_groups = inputs.shape[-1] // len(fmean)

        ## Repeat gammas
        gamma_f = jnp.concatenate(
            [
                jnp.tile(f, reps=len(t))
                for f, t in zip(
                    [gamma_f_a, gamma_f_t, gamma_f_d],
                    [gamma_theta_a, gamma_theta_t, gamma_theta_d],
                )
            ]
        )
        gamma_f = jnp.tile(gamma_f, reps=2)
        gamma_theta = jnp.concatenate(
            [
                jnp.tile(t, reps=len(f))
                for f, t in zip(
                    [gamma_f_a, gamma_f_t, gamma_f_d],
                    [gamma_theta_a, gamma_theta_t, gamma_theta_d],
                )
            ]
        )
        gamma_theta = jnp.tile(gamma_theta, reps=2)

        ## Repeating
        cc = jnp.array([0, 1, 2])
        cc = jnp.repeat(
            cc, repeats=jnp.array([64, 32, 32]), total_repeat_length=len(fmean)
        )

        kernel = jax.vmap(
            self.gaussian,
            in_axes=(None, None, 0, 0, 0, 0, None, 0, None, None),
            out_axes=1,
        )(fmean, theta_mean, fmean, theta_mean, gamma_f, gamma_theta, cc, cc, H_cc, 1)
        kernel = kernel[None, None, :, :]

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4:
            inputs = inputs[None, :]
            had_batch = False
        else:
            had_batch = True
        outputs = lax.conv_general_dilated(
            jnp.transpose(inputs, [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (self.strides, self.strides),
            self.padding,
        )
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0, 2, 3, 1))
        if not had_batch:
            outputs = outputs[0]
        return outputs + bias

    @staticmethod
    def gaussian(
        f, theta, fmean, theta_mean, gamma_f, gamma_theta, c_1, c_2, H_cc, A=1
    ):
        return (
            H_cc[c_1, c_2]
            * A
            * jnp.exp(-((gamma_f**2) * (f - fmean) ** 2) / (2))
            * jnp.exp(-((gamma_theta**2) * (theta - theta_mean) ** 2) / (2))
        )


class GDNSpatioChromaFreqOrient(nn.Module):
    """Generalized Divisive Normalization."""

    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    # inputs_star: float = 1.
    # outputs_star: Union[None, float] = None
    fs: int = 1
    apply_independently: bool = False
    bias_init: Callable = nn.initializers.ones_init()
    alpha: float = 2.0
    epsilon: float = 1 / 2  # Exponential of the denominator
    eps: float = 1e-6  # Numerical stability in the denominator
    normalize_prob: bool = False
    normalize_energy: bool = True

    @nn.compact
    def __call__(
        self,
        inputs,
        fmean,
        theta_mean,
        train=False,
    ):
        b, h, w, c = inputs.shape
        bias = self.param(
            "bias",
            # equal_to(inputs_star/10),
            self.bias_init,
            (c,),
        )
        # is_initialized = self.has_variable("batch_stats", "inputs_star")
        # inputs_star = self.variable("batch_stats", "inputs_star", lambda x: jnp.ones(x)*self.inputs_star, (len(self.inputs_star),))
        # inputs_star_ = jnp.ones_like(inputs)*inputs_star.value
        GL = GaussianLayerGamma(
            features=c,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="VALID",
            fs=self.fs,
            xmean=self.kernel_size / self.fs / 2,
            ymean=self.kernel_size / self.fs / 2,
            normalize_prob=self.normalize_prob,
            normalize_energy=self.normalize_energy,
            use_bias=False,
            feature_group_count=c,
        )
        FOG = ChromaFreqOrientGaussianGamma()
        outputs = GL(
            pad_same_from_kernel_size(
                inputs, kernel_size=self.kernel_size, mode=self.padding
            )
            ** self.alpha,
            train=train,
        )  # /(self.kernel_size**2)
        outputs = FOG(outputs, fmean=fmean, theta_mean=theta_mean)

        ## Coef
        # coef = GL(inputs_star_**self.alpha, train=train)#/(self.kernel_size**2)
        # coef = FG(coef, fmean=fmean)
        # coef = rearrange(coef, "b h w (phase theta f) -> b h w (phase f theta)", b=b, h=h, w=w, phase=2, f=config.N_SCALES, theta=config.N_ORIENTATIONS)
        # coef = OG(coef, theta_mean=theta_mean) + bias
        # coef = rearrange(coef, "b h w (phase f theta) -> b h w (phase theta f)", b=b, h=h, w=w, phase=2, f=config.N_SCALES, theta=config.N_ORIENTATIONS)
        # coef = jnp.clip(coef+bias, a_min=1e-5)**self.epsilon
        # # coef = inputs_star.value * coef
        # if self.outputs_star is not None: coef = coef/inputs_star.value*self.outputs_star

        # if is_initialized and train:
        #     inputs_star.value = (inputs_star.value + jnp.quantile(jnp.abs(inputs), q=0.95, axis=(0,1,2)))/2
        # return coef * inputs / (jnp.clip(denom+bias, a_min=1e-5)**self.epsilon + self.eps)
        return (
            self.param("B", nn.initializers.ones_init(), (outputs.shape[-1],))
            * inputs
            / (jnp.clip(outputs + bias, a_min=1e-5) ** self.epsilon + self.eps)
        )


class GaborLayerGammaHumanLike_(nn.Module):
    """Parametric Gabor layer with particular initialization."""

    n_scales: Sequence[int]  # [A, T, D]
    n_orientations: Sequence[int]  # [A, T, D]

    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1

    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1  # Sampling frequency
    phase = jnp.array([0.0, jnp.pi / 2.0])

    normalize_prob: bool = True
    normalize_energy: bool = False
    zero_mean: bool = False
    train_A: bool = False

    @nn.compact
    def __call__(
        self,
        inputs,
        train=False,
        return_freq=False,
        return_theta=False,
    ):
        total_scales = jnp.sum(jnp.array(self.n_scales))
        total_orientations = jnp.sum(jnp.array(self.n_orientations))
        features = jnp.sum(
            jnp.array(
                [
                    s * o * len(self.phase)
                    for s, o in zip(self.n_scales, self.n_orientations)
                ]
            )
        )

        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable(
            "precalc_filter",
            "kernel",
            jnp.zeros,
            (self.kernel_size, self.kernel_size, inputs.shape[-1], features),
        )
        freq_a = self.param(
            "freq_a",
            freq_scales_init(n_scales=self.n_scales[0], fs=self.fs),
            (self.n_scales[0],),
        )
        gammax_a = self.param(
            "gammax_a", k_array(k=0.4, arr=1 / (freq_a**0.8)), (self.n_scales[0],)
        )
        gammay_a = self.param("gammay_a", equal_to(gammax_a * 0.8), (self.n_scales[0],))
        theta_a = self.param(
            "theta_a",
            linspace(start=0, stop=jnp.pi, num=self.n_orientations[0]),
            (self.n_orientations[0],),
        )
        sigma_theta_a = self.param(
            "sigma_theta_a", equal_to(theta_a), (self.n_orientations[0],)
        )

        freq_t = self.param(
            "freq_t",
            freq_scales_init(n_scales=self.n_scales[1], fs=self.fs),
            (self.n_scales[1],),
        )
        gammax_t = self.param(
            "gammax_t", k_array(k=0.4, arr=1 / (freq_t**0.8)), (self.n_scales[1],)
        )
        gammay_t = self.param("gammay_t", equal_to(gammax_t * 0.8), (self.n_scales[1],))
        theta_t = self.param(
            "theta_t",
            linspace(start=0, stop=jnp.pi, num=self.n_orientations[1]),
            (self.n_orientations[1],),
        )
        sigma_theta_t = self.param(
            "sigma_theta_t", equal_to(theta_t), (self.n_orientations[1],)
        )

        freq_d = self.param(
            "freq_d",
            freq_scales_init(n_scales=self.n_scales[2], fs=self.fs),
            (self.n_scales[2],),
        )
        gammax_d = self.param(
            "gammax_d", k_array(k=0.4, arr=1 / (freq_d**0.8)), (self.n_scales[2],)
        )
        gammay_d = self.param("gammay_d", equal_to(gammax_d * 0.8), (self.n_scales[2],))
        theta_d = self.param(
            "theta_d",
            linspace(start=0, stop=jnp.pi, num=self.n_orientations[2]),
            (self.n_orientations[2],),
        )
        sigma_theta_d = self.param(
            "sigma_theta_d", equal_to(theta_d), (self.n_orientations[2],)
        )

        A = self.param("A", nn.initializers.ones_init(), (inputs.shape[-1], 128))
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (features,))
        else:
            bias = 0.0

        if is_initialized and not train:
            kernel = precalc_filters.value
        elif is_initialized and train:
            x, y = self.generate_dominion()
            ## A
            kernel_a = jax.vmap(
                self.gabor,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )
            kernel_a = jax.vmap(
                kernel_a,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )
            kernel_a = jax.vmap(
                kernel_a,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )(
                x,
                y,
                self.xmean,
                self.ymean,
                gammax_a,
                gammay_a,
                freq_a,
                theta_a,
                sigma_theta_a,
                self.phase,
                1,
                self.normalize_prob,
                self.normalize_energy,
                self.zero_mean,
            )
            kernel_a = rearrange(
                kernel_a, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)"
            )
            kernel_a = repeat(
                kernel_a,
                "kx ky c_out -> kx ky c_in c_out",
                c_in=inputs.shape[-1],
                c_out=kernel_a.shape[-1],
            )

            ## T
            kernel_t = jax.vmap(
                self.gabor,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )
            kernel_t = jax.vmap(
                kernel_t,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )
            kernel_t = jax.vmap(
                kernel_t,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )(
                x,
                y,
                self.xmean,
                self.ymean,
                gammax_t,
                gammay_t,
                freq_t,
                theta_t,
                sigma_theta_t,
                self.phase,
                1,
                self.normalize_prob,
                self.normalize_energy,
                self.zero_mean,
            )
            kernel_t = rearrange(
                kernel_t, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)"
            )
            kernel_t = repeat(
                kernel_t,
                "kx ky c_out -> kx ky c_in c_out",
                c_in=inputs.shape[-1],
                c_out=kernel_t.shape[-1],
            )

            ## D
            kernel_d = jax.vmap(
                self.gabor,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )
            kernel_d = jax.vmap(
                kernel_d,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )
            kernel_d = jax.vmap(
                kernel_d,
                in_axes=(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                ),
                out_axes=0,
            )(
                x,
                y,
                self.xmean,
                self.ymean,
                gammax_d,
                gammay_d,
                freq_d,
                theta_d,
                sigma_theta_d,
                self.phase,
                1,
                self.normalize_prob,
                self.normalize_energy,
                self.zero_mean,
            )
            kernel_d = rearrange(
                kernel_d, "phases rots fs_sigmas kx ky -> kx ky (phases rots fs_sigmas)"
            )
            kernel_d = repeat(
                kernel_d,
                "kx ky c_out -> kx ky c_in c_out",
                c_in=inputs.shape[-1],
                c_out=kernel_d.shape[-1],
            )

            ## Concat all of them
            kernel = jnp.concatenate([kernel_a, kernel_t, kernel_d], axis=-1)
            kernel = kernel * A[None, None, :, :]
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4:
            inputs = inputs[None, :]
            had_batch = False
        else:
            had_batch = True
        outputs = lax.conv(
            jnp.transpose(inputs, [0, 3, 1, 2]),  # lhs = NCHW image tensor
            jnp.transpose(kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
            (self.strides, self.strides),
            self.padding,
        )
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0, 2, 3, 1))
        fmean = jnp.concatenate(
            [
                jnp.tile(f, reps=len(t))
                for f, t in zip([freq_a, freq_t, freq_d], [theta_a, theta_t, theta_d])
            ]
        )
        fmean = jnp.tile(fmean, reps=2)
        theta_mean = jnp.concatenate(
            [
                jnp.tile(t, reps=len(f))
                for f, t in zip([freq_a, freq_t, freq_d], [theta_a, theta_t, theta_d])
            ]
        )
        theta_mean = jnp.tile(theta_mean, reps=2)

        if not had_batch:
            outputs = outputs[0]
        if return_freq and return_theta:
            return outputs + bias, fmean, theta_mean
        elif return_freq and not return_theta:
            return outputs + bias, fmean
        elif not return_freq and return_theta:
            return outputs + bias, theta_mean
        else:
            return outputs + bias

    @staticmethod
    def gabor(
        x,
        y,
        xmean,
        ymean,
        gammax,
        gammay,
        freq,
        theta,
        sigma_theta,
        phase,
        A=1,
        normalize_prob=True,
        normalize_energy=False,
        zero_mean=False,
    ):
        x, y = x - xmean, y - ymean
        ## Obtain the normalization coeficient
        gamma_vector = jnp.array([gammax, gammay])
        inv_cov_matrix = jnp.diag(gamma_vector) ** 2
        # det_cov_matrix = 1/jnp.linalg.det(cov_matrix)
        # # A_norm = 1/(2*jnp.pi*jnp.sqrt(det_cov_matrix)) if normalize_prob else 1.
        # A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*jnp.sqrt(det_cov_matrix)), 1.)
        A_norm = 1.0

        ## Rotate the sinusoid
        rotation_matrix = jnp.array(
            [
                [jnp.cos(sigma_theta), -jnp.sin(sigma_theta)],
                [jnp.sin(sigma_theta), jnp.cos(sigma_theta)],
            ]
        )
        rotated_covariance = (
            rotation_matrix @ inv_cov_matrix @ jnp.transpose(rotation_matrix)
        )
        x_r_1 = rotated_covariance[0, 0] * x + rotated_covariance[0, 1] * y
        y_r_1 = rotated_covariance[1, 0] * x + rotated_covariance[1, 1] * y
        distance = x * x_r_1 + y * y_r_1
        g = (
            A_norm
            * jnp.exp(-distance / 2)
            * jnp.cos(
                2 * jnp.pi * freq * (x * jnp.cos(theta) + y * jnp.sin(theta)) + phase
            )
        )
        g = jnp.where(zero_mean, g - g.mean(), g)
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(g**2)), 1.0)
        return A * g / E_norm

    def return_kernel(self, params, c_in=3):
        x, y = self.generate_dominion()
        sigmax, sigmay = jnp.exp(params["sigmax"]), jnp.exp(params["sigmay"])
        kernel = jax.vmap(
            self.gabor,
            in_axes=(
                None,
                None,
                None,
                None,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            out_axes=0,
        )
        kernel = jax.vmap(
            kernel,
            in_axes=(
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            out_axes=0,
        )
        kernel = jax.vmap(
            kernel,
            in_axes=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0,
                0,
                None,
                None,
                None,
            ),
            out_axes=0,
        )(
            x,
            y,
            self.xmean,
            self.ymean,
            params["sigmax"],
            params["sigmay"],
            params["freq"],
            params["theta"],
            params["sigma_theta"],
            self.phase,
            1,
            self.normalize_prob,
            self.normalize_energy,
        )
        # kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
        kernel = rearrange(kernel, "rots fs sigmas kx ky -> kx ky (rots fs sigmas)")
        kernel = repeat(
            kernel, "kx ky c_out -> kx ky c_in c_out", c_in=c_in, c_out=kernel.shape[-1]
        )
        return kernel

    def generate_dominion(self):
        return jnp.meshgrid(
            jnp.linspace(0, self.kernel_size / self.fs, num=self.kernel_size),
            jnp.linspace(0, self.kernel_size / self.fs, num=self.kernel_size),
        )
