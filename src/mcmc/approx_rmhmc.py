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
    dim = x_init.shape[0]

    @jax.jit
    def neg_log_prob(x: jax.Array) -> float:
        """Negative log-probability (potential energy)"""
        return -log_prob(x)

    # Compute gradient of the negative log-probability
    grad_nll = jax.grad(neg_log_prob)

    @jax.jit
    def __hamiltonian(
        x: jax.Array,
        p: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the Hamiltonian and its gradient with respect to position x"""

        grad_val = grad_nll(x)
        metric_fn = lambda u: grad_val * (grad_val @ u) + u * 1e-8

        # Compute the dot products with the inverse of the Riemannian metric tensor
        v = jax.scipy.sparse.linalg.cg(metric_fn, grad_val, maxiter=dim, tol=1e-6)[0]
        w = jax.scipy.sparse.linalg.cg(metric_fn, p, maxiter=dim, tol=1e-6)[0]

        # Total Hamiltonian H = 1/2 p^T G^{-1}(x) p + 1/2 log|G(x)| + U(x)
        hamiltonian = 0.5 * p.T @ w + 0.5 * jnp.log(1) + neg_log_prob(x)

        # Partial derivative of the Hamiltonian w.r.t position x
        grad_hamiltonian = (
            grad_val + (grad_val @ v) * grad_val + (p @ v) * (grad_val @ w) * grad_val
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
        x, p, grad_val, w, w_new = carry
        x_new = x + 0.5 * eps * (w + w_new)

        metric_fn = lambda u: grad_val * (grad_val @ u) + u * 1e-8
        w_new = jax.scipy.sparse.linalg.cg(metric_fn, p, maxiter=dim, tol=1e-6)[0]

        return (x, p, grad_val, w, w_new), x_new

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

        grad_val = grad_nll(x)
        p = fp_arr_p[-1]

        metric_fn = lambda u: grad_val * (grad_val @ u) + u * 1e-8
        w = jax.scipy.sparse.linalg.cg(metric_fn, p, maxiter=dim, tol=1e-6)[0]

        # Update position implicitly (full step)
        carry, fp_arr_x = jax.lax.scan(
            fp_x,
            (x, p, grad_val, w, w),
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
        (
            key,
            rej,
            x,
        ) = carry
        key, subkey0, subkey1 = jax.random.split(key, 3)

        # Sample initial momentum from N(0, G(x))
        p = jax.random.normal(subkey0, x.shape)
        grad_val = grad_nll(x)
        p = grad_val * (grad_val @ p)

        # Compute initial Hamiltonian
        hamiltonian, grad_hamiltonian = __hamiltonian(x, p)

        # Run generalized leapfrog integrator
        leap_carry, leap = jax.lax.scan(
            leapfrog,
            (
                x,
                p,
                hamiltonian,
            ),
            None,
            t_max,
        )

        (
            x_new,
            _,
            new_hamiltonian,
        ) = leap_carry

        # Accept-reject step
        u = jax.random.uniform(subkey1)
        condition = u < jnp.exp(hamiltonian - new_hamiltonian)

        new_rej = jax.lax.select(condition, rej, rej + 1)
        new_x = jax.lax.select(condition, x_new, x)

        output = leap if return_path else new_x
        return (
            key,
            new_rej,
            new_x,
        ), output

    carry, samples = jax.lax.scan(
        _loop,
        (
            key,
            0,
            x_init,
        ),
        None,
        n_iter,
    )

    _, rej, _, _, _, _, _, _ = carry
    rejection_rate = rej / n_iter

    if return_path:
        samples = samples.reshape(-1, x_init.shape[-1])

    return rejection_rate, samples
