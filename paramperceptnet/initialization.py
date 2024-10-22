from jax import numpy as jnp


def humanlike_init(params):
    params_ = params.copy()
    params_["GDNGamma_0"]["bias"] = jnp.ones_like(params_["GDNGamma_0"]["bias"]) * 0.1
    params_["GDNGamma_0"]["kernel"] = (
        jnp.ones_like(params_["GDNGamma_0"]["kernel"]) * 0.5
    )

    ## Opponent color channel transformation
    params_["Color"]["kernel"] = (
        jnp.array(
            [
                [39.0454, 30.1207, 14.27948],
                [115.8404, -63.3502, 41.26816],
                [16.3118, 30.2934, -61.51888],
            ]
        )[None, None, :, :]
        / 163.5217
    )

    ## Center Surround
    params_["CenterSurroundLogSigmaK_0"]["logsigma"] = jnp.array(
        [-1.9, -1.9, -1.9, -1.76, -1.76, -1.76, -1.76, -1.76, -1.76]
    )
    params_["CenterSurroundLogSigmaK_0"]["K"] = jnp.array(
        [1.1, 1.1, 1.1, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    )
    params_["CenterSurroundLogSigmaK_0"]["A"] = jnp.array(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )

    ## GDNGaussian
    params_["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"] = jnp.ones_like(
        params_["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"]
    ) * (1.0 / 0.04)
    params_["GDNGaussian_0"]["GaussianLayerGamma_0"]["bias"] = (
        jnp.ones_like(params_["GDNGaussian_0"]["GaussianLayerGamma_0"]["bias"]) * 0.1
    )

    ## Gabor
    params_["GaborLayerGammaHumanLike__0"]["freq_a"] = jnp.array([2.0, 4.0, 8.0, 16.0])
    params_["GaborLayerGammaHumanLike__0"]["freq_t"] = jnp.array([3.0, 6.0])
    params_["GaborLayerGammaHumanLike__0"]["freq_d"] = jnp.array([3.0, 6.0])

    params_["GaborLayerGammaHumanLike__0"]["gammax_a"] = (
        params_["GaborLayerGammaHumanLike__0"]["freq_a"] ** 0.9
    )
    params_["GaborLayerGammaHumanLike__0"]["gammay_a"] = (
        0.8 * params_["GaborLayerGammaHumanLike__0"]["gammax_a"]
    )

    params_["GaborLayerGammaHumanLike__0"]["gammax_t"] = (
        params_["GaborLayerGammaHumanLike__0"]["freq_t"] ** 0.9
    )
    params_["GaborLayerGammaHumanLike__0"]["gammay_t"] = (
        0.8 * params_["GaborLayerGammaHumanLike__0"]["gammax_t"]
    )

    params_["GaborLayerGammaHumanLike__0"]["gammax_d"] = (
        params_["GaborLayerGammaHumanLike__0"]["freq_d"] ** 0.9
    )
    params_["GaborLayerGammaHumanLike__0"]["gammay_d"] = (
        0.8 * params_["GaborLayerGammaHumanLike__0"]["gammax_d"]
    )
    # params_["GaborLayerGammaHumanLike__0"]["theta_a"] = jnp.tile(jnp.linspace(0., jnp.pi, num=16), reps=128//16)
    # params_["GaborLayerGammaHumanLike__0"]["sigma_theta_a"] = params_["GaborLayerGammaHumanLike__0"]["theta_a"]
    # params_["GaborLayerGammaHumanLike__0"]["phase_a"] = jnp.repeat(jnp.array([0., 90.]), repeats=64)

    A_a = jnp.zeros(shape=(3, 64), dtype=jnp.float32)
    A_a = A_a.at[0, :].set(1.0)
    A_t = jnp.zeros(shape=(3, 32), dtype=jnp.float32)
    A_t = A_t.at[1, :].set(1.0)
    A_d = jnp.zeros(shape=(3, 32), dtype=jnp.float32)
    A_d = A_d.at[2, :].set(1.0)
    params_["GaborLayerGammaHumanLike__0"]["A"] = jnp.concatenate(
        [A_a, A_t, A_d], axis=-1
    )

    ## GDNSpatioChromaFreqOrient
    params_["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"] = (
        jnp.ones_like(
            params_["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"]
        )
        * (1.0 / 0.2)
    )
    # params_["GDNSpatioChromaFreqOrient_0"]["OrientGaussianGamma_0"]["gamma"] = jnp.ones_like(params_["GDNSpatioChromaFreqOrient_0"]["OrientGaussianGamma_0"]["gamma"])*(1/20)
    # params_["GDNSpatioChromaFreqOrient_0"]["bias"] = jnp.tile(jnp.array([0.001, 0.002, 0.0035, 0.01])/100, reps=config.N_ORIENTATIONS*2)
    params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
        "H_cc"
    ] = jnp.eye(3, 3)
    params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
        "gamma_theta_a"
    ] = jnp.ones_like(
        params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
            "gamma_theta_a"
        ]
    ) * (1 / (20 * jnp.pi / 180))
    params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
        "gamma_theta_t"
    ] = jnp.ones_like(
        params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
            "gamma_theta_t"
        ]
    ) * (1 / (20 * jnp.pi / 180))
    params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
        "gamma_theta_d"
    ] = jnp.ones_like(
        params_["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"][
            "gamma_theta_d"
        ]
    ) * (1 / (20 * jnp.pi / 180))
    return params_
