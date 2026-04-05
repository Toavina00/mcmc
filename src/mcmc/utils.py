import jax
import jax.numpy as jnp

def ess(samples: jax.Array) -> float:
    """
    Compute the Effective Sample Size

    :Parameters
        - samples: samples from a monte carlo sampler

    :Returns
        - ess: Effective Sample Size
    """

    samples -= samples.mean()

    power_sp = jnp.abs(jnp.fft.fft(samples)) ** 2
    autocorr = jnp.fft.ifft(power_sp)

    return len(samples) / autocorr.sum()