import jax
import jax.numpy as jnp

from typing import Callable, Tuple

def sample(
    key: jax.Array,
    prob: Callable[[jax.Array], float],
    x_init: jax.Array,
    n_iter: int,
    eps: float,
    tau: int,
) -> Tuple[float, jax.Array]:

    """
    Sample from a given probability distribution using Hamiltonian Monte Carlo

    :Parameters
        - key: jax random key
        - prob: probability density which we are sampling from
        - x_init: initial position
        - n_iter: number of iterations
        - eps: leapfrog step size
        - tau: leapfrog iteration
    
    :Returns
        - rejection_rate: sampling rejection rate
        - samples: samples obtained

    """

    E = lambda x: -jnp.log(prob(x))
    K = lambda p: (p.T @ p) / 2
    gE = jax.grad(E)

    def leapfrog(carry, _):
        x, p = carry
        p = p - 0.5 * eps * gE(x)
        x = x + eps * p
        p = p - 0.5 * eps * gE(x)
        return (x, p), (x, p)

    def _loop(carry, _):
        _key, rej, x = carry
        _key, subkey0, subkey1 = jax.random.split(_key, 3)

        p = jax.random.normal(subkey0, x.shape)
        H = K(p) + E(x)

        _, leap = jax.lax.scan(leapfrog, (x, p), None, tau)

        x_new = leap[0][-1]
        p_new = leap[1][-1]

        H_new = K(p_new) + E(x_new)
        dH = H_new - H
        u = jax.random.uniform(subkey1)

        condition = jnp.logical_or(dH < 0, u < jnp.exp(-dH))
        new_rej = jax.lax.select(condition, rej, rej + 1)
        new_x = jax.lax.select(condition, x_new, x)

        return (_key, new_rej, new_x), new_x

    carry, samples = jax.lax.scan(_loop, (key, 0, x_init), None, n_iter)

    _, rej, _ = carry
    rejection_rate = rej / n_iter

    return rejection_rate, samples



def sample_with_path(
    key: jax.Array,
    prob: Callable[[jax.Array], float],
    x_init: jax.Array,
    n_iter: int,
    eps: float,
    tau: int,
) -> Tuple[float, jax.Array]:

    """
    Sample from a given probability distribution using Hamiltonian Monte Carlo and return the dynamics' path

    :Parameters
        - key: jax random key
        - prob: probability density which we are sampling from
        - x_init: initial position
        - n_iter: number of iterations
        - eps: leapfrog step size
        - tau: leapfrog iteration
    
    :Returns
        - rejection_rate: sampling rejection rate
        - path: dynamics' path

    """

    E = lambda x: -jnp.log(prob(x))
    K = lambda p: (p.T @ p) / 2
    gE = jax.grad(E)

    def leapfrog(carry, _):
        x, p = carry
        p = p - 0.5 * eps * gE(x)
        x = x + eps * p
        p = p - 0.5 * eps * gE(x)
        return (x, p), (x, p)

    def _loop(carry, _):
        _key, rej, x = carry
        _key, subkey0, subkey1 = jax.random.split(_key, 3)

        p = jax.random.normal(subkey0, x.shape)
        H = K(p) + E(x)

        _, leap = jax.lax.scan(leapfrog, (x, p), None, tau)

        x_new = leap[0][-1]
        p_new = leap[1][-1]

        H_new = K(p_new) + E(x_new)
        dH = H_new - H
        u = jax.random.uniform(subkey1)

        condition = jnp.logical_or(dH < 0, u < jnp.exp(-dH))
        new_rej = jax.lax.select(condition, rej, rej + 1)
        new_x = jax.lax.select(condition, x_new, x)

        return (_key, new_rej, new_x), leap[0]

    carry, path = jax.lax.scan(_loop, (key, 0, x_init), None, n_iter)

    _, rej, _ = carry
    rejection_rate = rej / n_iter

    path = path.reshape(-1, x_init.shape[-1])

    return rejection_rate, path