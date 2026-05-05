from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def sample(
    key: jax.Array,
    log_prob: Callable[[jax.Array], float],
    x_init: jax.Array,
    n_iter: int,
    eps: float,
    tau: int,
    mass_matrix: jax.Array | None = None,
    tk_reg: float = 1e-8,
    return_path: bool = False,
) -> Tuple[float | jax.Array, jax.Array]:
    """
    Sample from a given probability distribution using Hamiltonian Monte Carlo.

    :Parameters
        - key: jax random key
        - log_prob: log-probability density which we are sampling from
        - x_init: initial position
        - n_iter: number of iterations
        - eps: leapfrog step size
        - tau: leapfrog iterations
        - mass_matrix: the covariance matrix of the momentum `p`, set to the identity if `None`
        - tk_reg: Tikhonov regularization coefficient for stability
        - return_path: if `True, return the full leapfrog dynamics path instead
                       of just the accepted samples

    :Returns
        - rejection_rate: sampling rejection rate
        - samples: accepted samples (shape: [n_iter, dim]) if return_path=False,
                   or full leapfrog path (shape: [n_iter * tau, dim]) if return_path=True
    """

    covariance, cov_cholesky = None, None

    if mass_matrix is not None:
        if mass_matrix.ndim != 2:
            raise ValueError("Mass matrix should be a 2D array")

        if mass_matrix.shape[0] != mass_matrix.shape[1]:
            raise ValueError("Mass matrix should be a square matrix")

        covariance = mass_matrix + tk_reg * jnp.eye(mass_matrix.shape[0])
        cov_cholesky = jnp.linalg.cholesky(covariance)

    @jax.jit
    def neg_log_prob(x: jax.Array) -> float:
        """Negative log-probability (potential energy)"""
        return -log_prob(x)

    @jax.jit
    def kinetic_energy(p: jax.Array) -> jax.Array:
        """Kinetic energy of the Hamiltonian system"""
        g = p if mass_matrix is None else jnp.linalg.solve(covariance, p)
        return (p.T @ g) * 0.5

    # Gradient of the negative log-probability
    grad_nll = jax.grad(neg_log_prob)

    def leapfrog(carry, _):
        x, p = carry
        # Half step for momentum
        p = p - 0.5 * eps * grad_nll(x)
        # Full step for position
        g = p if mass_matrix is None else jnp.linalg.solve(covariance, p)
        x = x + eps * g
        # Half step for momentum
        p = p - 0.5 * eps * grad_nll(x)
        return (x, p), (x, p)

    def _loop(carry, _):
        _key, rej, x = carry
        _key, subkey0, subkey1 = jax.random.split(_key, 3)

        # Sample initial momentum from N(0, M)
        p = jax.random.normal(subkey0, x.shape)
        p = p if cov_cholesky is None else cov_cholesky @ p

        # Compute initial Hamiltonian
        hamiltonian = kinetic_energy(p) + neg_log_prob(x)

        # Run leapfrog integrator
        _, leap = jax.lax.scan(leapfrog, (x, p), None, tau)

        x_new = leap[0][-1]
        p_new = leap[1][-1]

        # Compute new Hamiltonian at proposed state
        new_hamiltonian = kinetic_energy(p_new) + neg_log_prob(x_new)
        dH = new_hamiltonian - hamiltonian

        # Draw uniform random variable
        u = jax.random.uniform(subkey1)

        # Accept-reject step
        condition = jnp.logical_or(dH < 0, u < jnp.exp(-dH))

        # Update rejection count and state
        new_rej = jax.lax.select(condition, rej, rej + 1)
        new_x = jax.lax.select(condition, x_new, x)

        output = leap[0] if return_path else new_x
        return (_key, new_rej, new_x), output

    carry, outputs = jax.lax.scan(_loop, (key, 0, x_init), None, n_iter)

    _, rej, _ = carry
    rejection_rate = rej / n_iter

    if return_path:
        outputs = outputs.reshape(-1, x_init.shape[-1])

    return rejection_rate, outputs
