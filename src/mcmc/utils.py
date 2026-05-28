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

    # Compute covariance matrix
    covar = jnp.atleast_2d(jnp.cov(samples, rowvar=False))
    _, logdet_c = jnp.linalg.slogdet(covar)

    if jnp.isneginf(logdet_c):
        raise ValueError("Sample covariance matrix is singular, mESS is undefined.")

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
    _, logdet_s = jnp.linalg.slogdet(sigma_mat)

    if jnp.isneginf(logdet_s):
        raise ValueError("Batch covariance matrix is singular, mESS is undefined.")

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


def bulk_ess(samples: jax.Array) -> jax.Array:
    """
    Compute the bulk Effective Sample Size (bulk-ESS).
    Measures ESS for the bulk of the distribution (excluding tails).

    Reference:
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
        Rank-normalization, folding, and localization
        An improved R-hat for assessing convergence of MCMC*.
        Bayesian Analysis, 16(2), 667-718.

    :Parameters
        - samples: samples from a monte carlo sampler (n_samples, d) or (n_samples,)

    :Returns
        - bulk_ess: bulk Effective Sample Size per component
    """
    if samples.ndim not in (1, 2):
        raise ValueError("The chain should be 1D or 2D.")

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n, d = samples.shape
    if n < 2:
        raise ValueError("Need at least two samples to compute bulk-ESS.")

    # Rank-normalize: convert to normal quantiles using Blom's continuity correction
    ranks = jnp.argsort(jnp.argsort(samples, axis=0), axis=0) + 1
    z = jax.scipy.special.erfinv(2.0 * (ranks - 0.375) / (n + 0.25) - 1.0) * jnp.sqrt(
        2.0
    )

    # Compute ESS on rank-normalized samples
    return ess(z)


def tail_ess(samples: jax.Array, quantiles: tuple = (0.05, 0.95)) -> jax.Array:
    """
    Compute the tail Effective Sample Size (tail-ESS).
    Measures ESS in the tails of the distribution at specified quantiles.

    Reference:
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
        Rank-normalization, folding, and localization
        An improved R-hat for assessing convergence of MCMC*.
        Bayesian Analysis, 16(2), 667-718.

    :Parameters
        - samples: samples from a monte carlo sampler (n_samples, d) or (n_samples,)
        - quantiles: tuple of quantiles to evaluate tail-ESS (default: (0.05, 0.95))

    :Returns
        - tail_ess: tail ESS per component, minimum over all quantiles
    """
    if samples.ndim not in (1, 2):
        raise ValueError("The chain should be 1D or 2D.")

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n, d = samples.shape
    if n < 2:
        raise ValueError("Need at least two samples to compute tail-ESS.")

    tail_ess_vals = []

    for q in quantiles:
        # Extract values below/above quantile
        q_val = jnp.quantile(samples, q, axis=0, keepdims=True)
        mask = samples <= q_val if q < 0.5 else samples >= q_val
        tail_samples = mask.astype(samples.dtype)
        tail_ess_vals.append(ess(tail_samples))

    # Return minimum tail-ESS across all quantiles
    return jnp.min(jnp.stack(tail_ess_vals), axis=0)


def rhat(chains: jax.Array) -> jax.Array:
    """
    Compute the rank-normalized potential scale reduction factor (R̂).
    Convergence diagnostic for multiple MCMC chains.

    Reference:
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
        Rank-normalization, folding, and localization
        An improved R-hat for assessing convergence of MCMC*.
        Bayesian Analysis, 16(2), 667-718.

    :Parameters
        - chains: array of shape (n_chains, n_samples_per_chain) or
                  (n_chains, n_samples_per_chain, d) for multivariate

    :Returns
        - rhat: potential scale reduction factor per component. Values < 1.01 indicate convergence.
    """
    if chains.ndim == 2:
        chains = chains.reshape(chains.shape[0], chains.shape[1], 1)
    elif chains.ndim != 3:
        raise ValueError(
            "chains should be (n_chains, n_samples_per_chain) or (n_chains, n_samples_per_chain, d)"
        )

    # Split each chain in half
    _, n_samples, _ = chains.shape
    n_half = n_samples // 2
    chains = jnp.concatenate([chains[:, :n_half, :], chains[:, -n_half:, :]], axis=0)
    n_chains, n_samples, d = chains.shape

    if n_chains < 2:
        raise ValueError("Need at least 2 chains to compute R̂.")
    if n_samples < 2:
        raise ValueError("Need at least 2 samples per chain to compute R̂.")

    # Reshape: (n_chains * n_samples, d)
    all_samples = chains.reshape(-1, d)

    # Rank-normalize using Blom's continuity correction
    ranks = jnp.argsort(jnp.argsort(all_samples, axis=0), axis=0) + 1
    z = jax.scipy.special.erfinv(
        2.0 * (ranks - 0.375) / (n_chains * n_samples + 0.25) - 1.0
    ) * jnp.sqrt(2.0)
    z = z.reshape(n_chains, n_samples, d)

    # Compute chain means and overall mean
    chain_means = jnp.mean(z, axis=1)  # (n_chains, d)
    overall_mean = jnp.mean(z, axis=(0, 1))  # (d,)

    # Between-chain variance
    b = (
        n_samples
        / (n_chains - 1)
        * jnp.sum((chain_means - overall_mean[None, :]) ** 2, axis=0)
    )

    # Within-chain variance per chain
    w_per_chain = jnp.sum((z - chain_means[:, None, :]) ** 2, axis=1) / (n_samples - 1)
    w = jnp.mean(w_per_chain, axis=0)

    # Estimate variance
    var_hat = ((n_samples - 1) / n_samples) * w + b / n_samples

    # Potential scale reduction factor
    rhat_val = jnp.where(w > 0, jnp.sqrt(var_hat / w), 1.0)

    return rhat_val


def multivariate_rhat(chains: jax.Array) -> jax.Array:
    """
    Compute the multivariate rank-normalized potential scale reduction factor.
    Extension of R̂ to multivariate case using inverse of the covariance matrix.

    :Parameters
        - chains: array of shape (n_chains, n_samples_per_chain, d) for multivariate samples

    :Returns
        - mrhat: multivariate R̂ (scalar). Values < 1.01 indicate convergence.
    """
    if chains.ndim != 3:
        raise ValueError("chains should be (n_chains, n_samples_per_chain, d)")

    # Split each chain in half
    _, n_samples, _ = chains.shape
    n_half = n_samples // 2
    chains = jnp.concatenate([chains[:, :n_half, :], chains[:, -n_half:, :]], axis=0)
    n_chains, n_samples, d = chains.shape

    if n_chains < 2:
        raise ValueError("Need at least 2 chains to compute multivariate R̂.")
    if n_samples < 2:
        raise ValueError("Need at least 2 samples per chain to compute multivariate R̂.")

    # Reshape: (n_chains * n_samples, d)
    all_samples = chains.reshape(-1, d)

    # Rank-normalize using Blom's continuity correction
    ranks = jnp.argsort(jnp.argsort(all_samples, axis=0), axis=0) + 1
    z = jax.scipy.special.erfinv(
        2.0 * (ranks - 0.375) / (n_chains * n_samples + 0.25) - 1.0
    ) * jnp.sqrt(2.0)
    z = z.reshape(n_chains, n_samples, d)

    # Compute chain means and overall mean
    chain_means = jnp.mean(z, axis=1)  # (n_chains, d)
    overall_mean = jnp.mean(z, axis=(0, 1))  # (d,)

    # Between-chain covariance
    diff = chain_means - overall_mean[None, :]  # (n_chains, d)
    B = (n_samples / (n_chains - 1)) * (diff.T @ diff)

    # Within-chain covariance per chain
    W_per_chain = jnp.array(
        [
            ((z[i] - chain_means[i]).T @ (z[i] - chain_means[i])) / (n_samples - 1)
            for i in range(n_chains)
        ]
    )
    W = jnp.mean(W_per_chain, axis=0)

    # Estimate covariance
    Sigma_hat = ((n_samples - 1) / n_samples) * W + B / n_samples

    # Inverse covariance with regularization
    W_reg = W + 1e-8 * jnp.trace(W) / d * jnp.eye(d)
    inv_W = jnp.linalg.inv(W_reg)

    mrhat_val = jnp.sqrt(jnp.trace(inv_W @ Sigma_hat) / d)
    # Clip to avoid NaN when variance is near-zero
    mrhat_val = jnp.where(jnp.isfinite(mrhat_val), mrhat_val, 1.0)

    return mrhat_val
