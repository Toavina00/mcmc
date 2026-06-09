from typing import Callable, Literal, Tuple

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
    mode: Literal["svd", "power_iter"] = "svd",
    tk_reg: float = 1e-3,
    n_power_iters: int = 10,
    return_path: bool = False,
) -> Tuple[float | jax.Array, jax.Array]:
    """
    Sample from a given probability distribution using Riemannian Manifold Hamiltonian Monte Carlo
    with rank one approximation of the metric tensor

    :Parameters
        - key: jax random key
        - log_prob: log-probability density which we are sampling from
        - x_init: initial position
        - n_iter: number of iterations
        - eps: leapfrog step size
        - t_max: leapfrog iteration
        - f_max: leapfrog fixed point iteration
        - tk_reg: Tikhonov regularisation coefficient
        - n_power_iters: Number of iteration in power iteration for rank-1 approximation
        - return_path: if True, return the full leapfrog dynamics path instead
                       of just the accepted samples

    :Returns
        - rejection_rate: sampling rejection rate
        - samples: accepted samples (shape: [n_iter, dim]) if return_path=False,
                   or full leapfrog path (shape: [n_iter * tau, dim]) if return_path=True

    """

    if mode not in ["svd", "power_iter"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'svd' or 'power_iter'.")

    dim = x_init.shape[0]

    @jax.jit
    def neg_log_prob(x: jax.Array) -> float:
        """Negative log-probability (potential energy)"""
        return -log_prob(x)

    @jax.jit
    def __metric(x: jax.Array) -> jax.Array:
        """Compute the vector for the rank one approximation of the metric"""

        if mode == "svd":
            # Compute the Hessian and its dominant eigenvector using SVD
            hessian = jax.hessian(neg_log_prob)(x)
            u, s, _ = jnp.linalg.svd(hessian, hermitian=True)
            top_eigenvector = u[:, 0]
            top_singular_value = s[0]

            return jnp.sqrt(top_singular_value) * top_eigenvector

        def hvp(v):
            return jax.jvp(jax.grad(neg_log_prob), (x,), (v,))[1]

        # Initialize a random vector
        v = jnp.ones(x.shape)
        v = v / jnp.linalg.norm(v)

        # Power iteration loop to find the dominant eigenvector
        def power_iter(_, v):
            v_next = hvp(v)
            return v_next / jnp.linalg.norm(v_next)

        v = jax.lax.fori_loop(0, n_power_iters, power_iter, v)
        top_eigenvalue = jnp.dot(v, hvp(v))

        return jnp.sqrt(jnp.abs(top_eigenvalue)) * v

    @jax.jit
    def metric_log_det(metric: jax.Array) -> jax.Array:
        """Compute the log determinant of the metric tensor"""
        norm_u = jnp.linalg.norm(metric)
        return jnp.log(tk_reg) * dim + jnp.log1p((norm_u**2) / tk_reg)

    @jax.jit
    def metric_inv_op(metric: jax.Array, v: jax.Array) -> jax.Array:
        """Compute the operation `G(x)^{-1} v = (lambda I + u u^t)^{-1} v`"""
        norm_u = jnp.linalg.norm(metric)
        return (1 / tk_reg) * (v - (1 / (tk_reg + norm_u**2)) * (metric @ v) * metric)

    @jax.jit
    def metric_sqrt_op(metric: jax.Array, v: jax.Array) -> jax.Array:
        """Compute the operation `G(x)^{1/2} v = (lambda I + u u^t)^{1/2} v`"""
        norm_u = jnp.linalg.norm(metric)
        return (
            jnp.sqrt(tk_reg) * v
            + (1 / (2 * jnp.sqrt(tk_reg) + norm_u**2)) * (metric @ v) * metric
        )

    @jax.jit
    def hamiltonian_fn(
        x: jax.Array,
        p: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the Hamiltonian with respect to position x"""
        metric = __metric(x)

        # H = 1/2 p^T G^{-1}(x) p + 1/2 log|G(x)| + NLL(x)
        return (
            0.5 * p.T @ metric_inv_op(metric, p)
            + 0.5 * metric_log_det(metric)
            + neg_log_prob(x)
        )

    grad_hamiltonian = jax.grad(hamiltonian_fn, argnums=0)

    def fp_p(carry, _):
        """Fixed point iteration step for momentum update"""
        x, p0, p = carry
        grad_H = grad_hamiltonian(x, p)
        p_new = p0 - 0.5 * eps * grad_H

        return (x, p0, p_new), p_new

    def fp_x(carry, _):
        """Fixed point iteration step for position update"""
        x0, p, w0, w = carry
        x_new = x0 + 0.5 * eps * (w0 + w)
        metric = __metric(x_new)
        w_new = metric_inv_op(metric, p)

        return (x0, p, w0, w_new), x_new

    def leapfrog(carry, _):
        """Generalized leapfrog integration step"""
        x, p = carry

        # Update momentum implicitly (half step)
        _, fp_arr_p = jax.lax.scan(
            fp_p,
            (x, p, p),
            None,
            f_max,
        )
        p = fp_arr_p[-1]

        # Update position implicitly (full step)
        metric = __metric(x)
        w = metric_inv_op(metric, p)
        carry, fp_arr_x = jax.lax.scan(
            fp_x,
            (x, p, w, w),
            None,
            f_max,
        )
        x = fp_arr_x[-1]

        # Final momentum explicit update (half step)
        grad_H = grad_hamiltonian(x, p)
        p = p - 0.5 * eps * grad_H

        return (x, p), x

    def _loop(carry, _):
        key, rej, x = carry
        key, subkey0, subkey1 = jax.random.split(key, 3)

        # Compute metric approximation
        metric = __metric(x)

        # Sample initial momentum from N(0, G(x))
        p = jax.random.normal(subkey0, (dim,))
        p = metric_sqrt_op(metric, p)

        # Compute initial Hamiltonian
        hamiltonian = hamiltonian_fn(x, p)

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
        new_hamiltonian = hamiltonian_fn(x_new, p_new)
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
