from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def sample(
    key: jax.Array,
    log_prob: Callable[[jax.Array], float],
    x_init: jax.Array,
    n_iter: int,
    eps: float,
    t_max: int,
    f_max: int,
    return_path: bool = False,
) -> Tuple[float, jax.Array]:
    """
    Sample from a given probability distribution using Riemannian Manifold Hamiltonian Monte Carlo

    :Parameters
        - key: jax random key
        - log_prob: log-probability density which we are sampling from
        - x_init: initial position
        - n_iter: number of iterations
        - eps: leapfrog step size
        - t_max: leapfrog iteration
        - f_max: leapfrog fixed point iteration
        - return_path: if True, return the full leapfrog dynamics path instead
                       of just the accepted samples

    :Returns
        - rejection_rate: sampling rejection rate
        - samples: accepted samples (shape: [n_iter, dim]) if return_path=False,
                   or full leapfrog path (shape: [n_iter * tau, dim]) if return_path=True

    """

    @jax.jit
    def neg_log_prob(x: jax.Array) -> float:
        """Negative log-probability (potential energy)"""
        return -log_prob(x)

    # Compute gradient of the negative log-probability
    grad_nll = jax.grad(neg_log_prob)

    @jax.jit
    def compute_metric(x: jax.Array) -> jax.Array:
        """Compute the metric tensor"""
        return jnp.square(grad_nll(x)) + 1e-8

    @jax.jit
    def inv_metric_op(x: jax.Array, u: jax.Array):
        """Compute the operation `G(x)^{-1} u`"""
        return u / compute_metric(x)

    @jax.jit
    def sample_momentum(key: jax.Array, x: jax.Array) -> jax.Array:
        """Sample from the momentum distribution"""
        p = jax.random.normal(key, x.shape)
        return jnp.sqrt(compute_metric(x)) * p

    @jax.jit
    def __hamiltonian(
        x: jax.Array,
        p: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the Hamiltonian and its gradient with respect to position x"""

        grad_val = grad_nll(x)
        jac_val = jax.jacfwd(grad_nll)(x)

        diag_metric = grad_val**2 + 1e-8
        jac_metric = 2 * jnp.einsum("i,jm->mij", grad_val, jac_val)
        metric_inv_dot_jac = jac_metric / diag_metric
        w = p / diag_metric

        # Total Hamiltonian H = 1/2 p^T G^{-1}(x) p + 1/2 log|G(x)| + U(x)
        hamiltonian = (
            0.5 * p.T @ w + 0.5 * jnp.sum(jnp.log(diag_metric)) + neg_log_prob(x)
        )

        # Partial derivative of the Hamiltonian w.r.t position x
        grad_hamiltonian = (
            grad_nll(x)
            + 0.5 * jnp.einsum("ijk,j,k->i", jac_metric, w, w)
            - 0.5 * jnp.einsum("ijj", metric_inv_dot_jac)
        )

        return hamiltonian, grad_hamiltonian

    def fp_p(carry, _):
        """Fixed point iteration step for momentum update"""
        x, p = carry
        _, grad_hamiltonian = __hamiltonian(x, p)
        p_new = p - 0.5 * eps * grad_hamiltonian

        return (x, p), p_new

    def fp_x(carry, _):
        """Fixed point iteration step for position update"""
        x, p, w, w_new = carry
        x_new = x + 0.5 * eps * (w + w_new)
        w_new = inv_metric_op(x_new, p)

        return (x, p, w, w_new), x_new

    def leapfrog(carry, _):
        """Generalized leapfrog integration step"""
        (x, p, _) = carry

        # Update momentum implicitly (half step)
        _, fp_arr_p = jax.lax.scan(
            fp_p,
            (x, p),
            None,
            f_max,
        )
        p = fp_arr_p[-1]

        # Update position implicitly (full step)
        w = inv_metric_op(x, p)
        carry, fp_arr_x = jax.lax.scan(
            fp_x,
            (x, p, w, w),
            None,
            f_max,
        )
        x = fp_arr_x[-1]

        # Final momentum explicit update (half step)
        _, grad_hamiltonian = __hamiltonian(x, p)
        p = p - 0.5 * eps * grad_hamiltonian
        hamiltonian, grad_hamiltonian = __hamiltonian(x, p)

        return (x, p, hamiltonian), x

    def _loop(carry, _):
        key, rej, x = carry
        key, subkey0, subkey1 = jax.random.split(key, 3)

        # Sample initial momentum from N(0, G(x))
        p = sample_momentum(subkey0, x)

        # Compute initial Hamiltonian
        hamiltonian, _ = __hamiltonian(x, p)

        # Run generalized leapfrog integrator
        leap_carry, leap = jax.lax.scan(
            leapfrog,
            (x, p, hamiltonian),
            None,
            t_max,
        )

        x_new, _, new_hamiltonian = leap_carry

        # Accept-reject step
        u = jax.random.uniform(subkey1)
        condition = u < jnp.exp(hamiltonian - new_hamiltonian)

        new_rej = jax.lax.select(condition, rej, rej + 1)
        new_x = jax.lax.select(condition, x_new, x)

        output = leap if return_path else new_x
        return (key, new_rej, new_x), output

    carry, samples = jax.lax.scan(
        _loop,
        (key, 0, x_init),
        None,
        n_iter,
    )

    _, rej, _ = carry
    rejection_rate = rej / n_iter

    if return_path:
        samples = samples.reshape(-1, x_init.shape[-1])

    return rejection_rate, samples
