import jax
import jax.numpy as jnp


def m_ess(samples: jax.Array) -> jax.Array:
    """
    Compute the multivariate Effective Sample Size

    Reference:
      Dootika Vats, James M Flegal, Galin L Jones,
      Multivariate output analysis for Markov chain Monte Carlo,
      Biometrika, Volume 106, Issue 2, June 2019, Pages 321-337,
      https://doi.org/10.1093/biomet/asz002

    :Parameters
        - samples: samples from a monte carlo sampler

    :Returns
        - m_ess: multivariate Effective Sample Size
    """

    if samples.ndim != 1 and samples.ndim != 2:
        raise ValueError("The chain should be 1D or 2D.")

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n, d = samples.shape

    jitter = 1e-6 * jnp.eye(d)

    # Compute covariance matrix
    covar = jnp.atleast_2d(jnp.cov(samples, rowvar=False))

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
    sigma_mat = jnp.atleast_2d((batch_size / (num_batches - 1)) * (diff.T @ diff))

    # Use slogdet to avoid numerical issues with determinant
    _, logdet_c = jnp.linalg.slogdet(covar + jitter)
    _, logdet_s = jnp.linalg.slogdet(sigma_mat + jitter)

    return jnp.clip(n * jnp.exp((logdet_c - logdet_s) / d), min=1.0)


def ess(samples: jax.Array) -> jax.Array:
    """
    Compute the component-wise/univariate Effective Sample Size using Geyer's initial
    positive monotone sequence (IPMS) estimator.

    Reference:
        Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
        Statistical Science, 7(4), 473-483.

    :Parameters
        - samples: samples from a monte carlo sampler

    :Returns
        - ess: component-wise/univariate Effective Sample Size
    """

    if samples.ndim != 1 and samples.ndim != 2:
        raise ValueError("The chain should be 1D or 2D.")

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n, d = samples.shape

    # Compute Linear autocovariance via FFT
    samples_centered = samples - samples.mean(axis=0)
    n_fft = 2 ** int(jnp.ceil(jnp.log2(2 * n - 1)))
    samples_fft = jnp.fft.rfft(samples_centered, n=n_fft, axis=0)
    power_spec = jnp.abs(samples_fft) ** 2
    autocov = jnp.fft.irfft(power_spec, axis=0)[:n]

    # Normalize to autocorrelation
    autocorr = autocov / autocov[0]

    # Pair consecutive lags: Gamma_m = rho(2m) + rho(2m+1)
    n_pairs = n // 2
    gamma = autocorr[: 2 * n_pairs].reshape(n_pairs, 2, d).sum(axis=1)

    # Truncate gamma
    mask = gamma > 0
    mask = jnp.cumprod(mask.astype(jnp.int32), axis=0)
    gamma_truncated = gamma * mask

    # Enforce monotonicity
    gamma_monotone = jax.lax.associative_scan(jnp.minimum, gamma_truncated)

    # Compute tau_int
    tau_int = -1.0 + 2.0 * gamma_monotone.sum(axis=0)
    tau_int = jnp.maximum(tau_int, 1.0)  # Ensure tau_int is at least 1

    return jnp.clip(n / tau_int, min=1.0)
