import jax
import jax.numpy as jnp

from typing import Callable, Tuple


def sample(
    key: jax.Array,
    log_prob: Callable[[jax.Array], float],
    x_init: jax.Array,
    n_iter: int,
    sigma: float,
) -> Tuple[float, jax.Array]:
    """
    Sample from a given probability distribution using Metropolis-Hastins with a gaussian proposal density

    :Parameters
        - key: jax random key
        - log_prob: log-probability density which we are sampling from
        - x_init: initial position
        - n_iter: number of iterations
        - sigma: stdev of the gaussian proposal density

    :Returns
        - rejection_rate: sampling rejection rate
        - samples: samples obtained

    """

    def _loop(carry, _):
        _key, rej, x = carry
        _key, subkey0, subkey1 = jax.random.split(_key, 3)

        # Propose a new state using a Gaussian distribution
        x_new = x + jax.random.normal(subkey0, shape=x.shape) * sigma
        # Draw a uniform random variable
        u = jax.random.uniform(subkey1)

        # Accept-reject condition
        condition = u <= jnp.exp(log_prob(x_new) - log_prob(x))

        # Update state and rejection count based on the condition
        new_x = jax.lax.select(condition, x_new, x)
        new_rej = jax.lax.select(condition, rej, rej + 1)

        return (_key, new_rej, new_x), new_x

    # Run the MCMC loop using jax.lax.scan
    carry, samples = jax.lax.scan(_loop, (key, 0, x_init), None, n_iter)

    _, rej, _ = carry
    rejection_rate = rej / n_iter

    return rejection_rate, samples
