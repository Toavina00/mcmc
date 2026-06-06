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
    tk_reg: float = 1e-8,
    softabs: bool = False,
    softabs_alpha: float = 1e-2,
    return_path: bool = False,
) -> Tuple[float | jax.Array, jax.Array]:
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
        - softabs: if True, use the SoftAbs metric instead of the Hessian
        - softabs_alpha: the alpha parameter for the SoftAbs metric
        - tk_reg: Tikhonov regularization coefficient for stability
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

    # Compute gradient, and Hessian of the negative log-probability
    hessian_nll = jax.hessian(neg_log_prob)

    @jax.jit
    def __metric(x: jax.Array) -> jax.Array:
        """Compute the metric tensor G(x)"""
        hessian = hessian_nll(x)
        if softabs:
            # SoftAbs metric: G = U SoftAbs(D) U^T, where H = U D U^T is the eigendecomposition of the Hessian
            eigvals, eigvecs = jnp.linalg.eigh(hessian)
            softabs_eigvals = jnp.where(
                jnp.abs(eigvals) < 1e-6,
                1e-6,
                eigvals / jnp.tanh(softabs_alpha * eigvals),
            )
            metric = (eigvecs * softabs_eigvals) @ eigvecs.T
        else:
            # Standard metric: G = H
            metric = hessian
        return metric

    @jax.jit
    def __cholesky_metric(
        x: jax.Array,
    ) -> jax.Array:
        """Compute the Cholesky decomposition of the metric tensor G(x)"""
        metric = __metric(x)
        cholesky_metric = jnp.linalg.cholesky(
            metric + tk_reg * jnp.eye(metric.shape[0])
        )
        return cholesky_metric

    @jax.jit
    def __back_solve(x: jax.Array, p: jax.Array) -> jax.Array:
        """Compute G(x)^{-1}p"""
        cholesky_metric = __cholesky_metric(x)
        w = jax.scipy.linalg.cho_solve((cholesky_metric, True), p)
        return w

    @jax.jit
    def __hamiltonian(
        x: jax.Array,
        p: jax.Array,
    ) -> jax.Array:
        # Compute kinetic energy K = 1/2 p^T G(x)^{-1} p + 1/2 log|G(x)|
        cholesky_metric = __cholesky_metric(x)
        w = jax.scipy.linalg.cho_solve((cholesky_metric, True), p)
        log_det_metric = 2 * jnp.sum(jnp.log(jnp.diag(cholesky_metric)))
        K = 0.5 * p.T @ w + 0.5 * log_det_metric
        # Total Hamiltonian H = K + U
        hamiltonian = K + neg_log_prob(x)
        return hamiltonian

    grad_hamiltonian = jax.grad(__hamiltonian, argnums=0)

    def fp_p(_, val):
        """Fixed point iteration step for momentum update"""
        p_old, x, p = val
        grad_H = grad_hamiltonian(x, p_old)
        p_new = p - 0.5 * eps * grad_H
        return (p_new, x, p)

    def fp_x(_, val):
        """Fixed point iteration step for position update"""
        _, x, p, w, w_new = val
        x_new = x + 0.5 * eps * (w + w_new)
        w_new = __back_solve(x_new, p)
        return (x_new, x, p, w, w_new)

    def leapfrog(carry, _):
        """Generalized leapfrog integration step"""
        x, p = carry

        # Update momentum implicitly (half step)
        fp_out_p = jax.lax.fori_loop(
            0,
            f_max,
            fp_p,
            (p, x, p),
        )
        p = fp_out_p[0]
        w = __back_solve(x, p)

        # Update position implicitly (full step)
        fp_out_x = jax.lax.fori_loop(
            0,
            f_max,
            fp_x,
            (x, x, p, w, w),
        )
        x = fp_out_x[0]

        # Final momentum explicit update (half step)
        grad_H = grad_hamiltonian(x, p)
        p = p - 0.5 * eps * grad_H

        return (x, p), x

    def _loop(carry, _):
        key, rej, x = carry
        key, subkey0, subkey1 = jax.random.split(key, 3)

        # Sample initial momentum from N(0, G(x))
        cholesky_metric = __cholesky_metric(x)
        p = jax.random.normal(subkey0, x.shape)
        p = cholesky_metric @ p

        # Compute initial Hamiltonian
        hamiltonian = __hamiltonian(x, p)

        # Run generalized leapfrog integrator
        leap_carry, leap = jax.lax.scan(
            leapfrog,
            (x, p),
            None,
            t_max,
        )

        x_new, p_new = leap_carry

        # Accept-reject step
        u = jax.random.uniform(subkey1)
        new_hamiltonian = __hamiltonian(x_new, p_new)
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

    rej = carry[1]
    rejection_rate = rej / n_iter

    if return_path:
        samples = samples.reshape(-1, x_init.shape[-1])

    return rejection_rate, samples
