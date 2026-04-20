from typing import Literal

import jax
import jax.numpy as jnp


def m_ess(
    samples: jax.Array, method: Literal["spectral", "batchmeans"] = "spectral", **kwargs
) -> jax.Array:
    """
    Compute the multivariate Effective Sample Size

    :Parameters
        - samples: samples from a monte carlo sampler
        - method: `spectral` or `batchmeans`

    :Returns
        - m_ess: multivariate Effective Sample Size
    """

    if samples.ndim != 1 and samples.ndim != 2:
        raise ValueError("The chain should be 1D or 2D.")

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    match method:
        case "spectral":
            return __spectral_mess(samples)

        case "batchmeans":
            return __batchmeans_mess(samples)

        case _:
            raise ValueError("Unknown method")


def __spectral_mess(samples: jax.Array) -> jax.Array:
    """
    Compute the multivariate Effective Sample Size using spectral variance
    """

    n, d = samples.shape

    # Compute covariance matrix
    covar = jnp.cov(samples, rowvar=False)
    det_covar = jnp.linalg.det(covar)

    # Compute spectral
    samples_centered = samples - samples.mean(axis=0)
    samples_fft = jnp.fft.fft(samples_centered, n=2*n, axis=0)[:n]
    spectral_mat = jnp.einsum("mp,mk->mpk", samples_fft, samples_fft.conj()) / n

    # Kernel smoothing (Bartlett window)
    m = jnp.floor(jnp.sqrt(n)).astype(int)
    k = jnp.arange(-m, m + 1)
    weights = jnp.where(k == 0, 0.0, 1 - jnp.abs(k) / (m + 1))
    weights = weights / weights.sum()
    indices = k % n
    psd_at_zero = jnp.sum(spectral_mat[indices] * weights[:, None, None], axis=0)

    det_sigma = jnp.linalg.det(psd_at_zero.real)

    return n * (det_covar / det_sigma) ** (1 / d)


def __batchmeans_mess(samples: jax.Array) -> jax.Array:
    """
    Compute the multivariate Effective Sample Size using batch means

    Reference:
      Dootika Vats, James M Flegal, Galin L Jones,
      Multivariate output analysis for Markov chain Monte Carlo,
      Biometrika, Volume 106, Issue 2, June 2019, Pages 321-337,
      https://doi.org/10.1093/biomet/asz002
    """

    n, d = samples.shape

    # Compute covariance matrix
    covar = jnp.cov(samples, rowvar=False)
    det_covar = jnp.linalg.det(covar)

    # Reshape to (num_batches, batch_size, d)
    batch_size = jnp.floor(jnp.sqrt(n)).astype(int)
    num_batches = n // batch_size
    reshaped_samples = samples[: num_batches * batch_size].reshape(
        num_batches, batch_size, d
    )

    # Calculate means of each batch
    batch_means = jnp.mean(reshaped_samples, axis=1)
    overall_mean = jnp.mean(samples, axis=0)

    # Calculate Sigma: The variance of the batch means scaled by batch size
    diff = batch_means - overall_mean
    sigma_mat = (batch_size / (num_batches - 1)) * (diff.T @ diff)
    det_sigma = jnp.linalg.det(sigma_mat)

    return n * (det_covar / det_sigma) ** (1 / d)


def ess(
    samples: jax.Array, method: Literal["spectral", "batchmeans", "geyer"] = "geyer", **kwargs
) -> jax.Array:
    """
    Compute the component-wise/univariate Effective Sample Size

    :Parameters
        - samples: samples from a monte carlo sampler
        - method: `spectral`, `batchmeans` or `ips`

    :Returns
        - ess: component-wise/univariate Effective Sample Size
    """

    if samples.ndim != 1 and samples.ndim != 2:
        raise ValueError("The chain should be 1D or 2D.")

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    match method:
        case "spectral":
            return __spectral_ess(samples)

        case "batchmeans":
            return __batchmeans_ess(samples)

        case "geyer":
            return __geyer_ess(samples)

        case _:
            raise ValueError("Unknown method")


def __spectral_ess(samples: jax.Array) -> jax.Array:
    """
    Compute the univariate Effective Sample Size using spectral variance
    """

    n, d = samples.shape

    # Component-wise variance
    var = jnp.var(samples, axis=0)

    # Compute the auto-covariance sum
    samples_centered = samples - samples.mean(axis=0)
    samples_fft = jnp.fft.fft(samples_centered, n=2*n, axis=0)[:n]
    power_spec = jnp.abs(samples_fft) ** 2 / n

    # Variance estimation smoothing (Bartlett window)
    m = jnp.floor(jnp.sqrt(n)).astype(int)
    k = jnp.arange(-m, m + 1)
    weights = jnp.where(k == 0, 0.0, 1 - jnp.abs(k) / (m + 1))
    weights = weights / weights.sum()
    indices = k % n
    psd_at_zero = jnp.sum(power_spec[indices] * weights[:, None], axis=0)

    return n * var / psd_at_zero


def __batchmeans_ess(samples: jax.Array) -> jax.Array:
    """
    Compute the univariate Effective Sample Size using batch means
    """

    n, d = samples.shape

    # Component-wise variance
    var = jnp.var(samples, axis=0)

    # Reshape to (num_batches, batch_size, d)
    batch_size = jnp.floor(jnp.sqrt(n)).astype(int)
    num_batches = n // batch_size
    reshaped_samples = samples[: num_batches * batch_size].reshape(
        num_batches, batch_size, d
    )

    # Calculate means of each batch
    batch_means = jnp.mean(reshaped_samples, axis=1)
    overall_mean = jnp.mean(samples, axis=0)

    # Calculate Sigma: The variance of the batch means scaled by batch size
    diff = batch_means - overall_mean
    sigma = (batch_size / (num_batches - 1)) * jnp.sum(diff**2, axis=0)

    return n * var / sigma


def __geyer_ess(samples: jax.Array) -> jax.Array:
    """
    Compute the univariate Effective Sample Size using Geyer's initial
    positive sequence (IPS) estimator.

    Reference:
        Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
        Statistical Science, 7(4), 473-483.
    """

    n, d = samples.shape

    # Compute Linear autocovariance via FFT
    samples_centered = samples - samples.mean(axis=0)
    samples_fft = jnp.fft.fft(samples_centered, n=2*n, axis=0)
    power_spec = jnp.abs(samples_fft) ** 2
    autocov = jnp.fft.ifft(power_spec, axis=0).real[:n] / n

    # Normalize to autocorrelation
    autocorr = autocov / autocov[0]

    # Pair consecutive lags: Gamma_m = rho(2m) + rho(2m+1)
    n_pairs = n // 2
    gamma = autocorr[: 2 * n_pairs].reshape(n_pairs, 2, d).sum(axis=1)

    # Truncate gamma
    is_positive = gamma > 0
    mask = jnp.cumprod(is_positive.astype(jnp.int32), axis=0)
    truncated_gamma = gamma * mask

    # IAT = -1 + 2 * sum_m Gamma_m
    iat = -1.0 + 2.0 * truncated_gamma.sum(axis=0)

    return n / iat
