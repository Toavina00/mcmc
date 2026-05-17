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
    grad_nll = jax.grad(neg_log_prob)
    hessian_nll = jax.hessian(neg_log_prob)
    # Jacobain of the Hessian gives the derivatives of the metric tensor
    jac_hessian_nll = jax.jacfwd(hessian_nll)

    @jax.jit
    def cholesky_solve(lower_cholesky: jax.Array, x: jax.Array) -> jax.Array:
        out = jax.scipy.linalg.solve_triangular(lower_cholesky, x, lower=True)
        out = jax.scipy.linalg.solve_triangular(
            lower_cholesky, out, lower=True, trans="T"
        )
        return out

    @jax.jit
    def __riemann_metric(
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # Compute the Riemannian metric tensor G(x) (using the Hessian)
        metric = hessian_nll(x)
        # Cholesky decomposition of G(x)
        cholesky_metric = jnp.linalg.cholesky(
            metric + tk_reg * jnp.eye(metric.shape[0])
        )
        # Determinant of G(x)
        det_metric = jnp.square(jnp.prod(jnp.diag(cholesky_metric)))
        # Jacobian of G(x)
        jac_metric = jac_hessian_nll(x)
        # Jacobian of |G(x)|
        jac_det_metric = jax.vmap(
            lambda b: cholesky_solve(cholesky_metric, b), in_axes=2
        )(jac_metric)
        jac_det_metric = jnp.einsum("jji", jac_det_metric)
        return cholesky_metric, det_metric, jac_metric, jac_det_metric

    @jax.jit
    def __hamiltonian(
        x: jax.Array,
        p: jax.Array,
        cholesky_metric: jax.Array,
        det_metric: jax.Array,
        jac_metric: jax.Array,
        jac_det_metric: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # Compute kinetic energy K = 1/2 p^T G(x)^{-1} p + 1/2 log|G(x)|
        w = cholesky_solve(cholesky_metric, p)
        K = 0.5 * p.T @ w + 0.5 * jnp.log(det_metric)
        # Total Hamiltonian H = K + U
        hamiltonian = K + neg_log_prob(x)
        # Partial derivative of the Hamiltonian w.r.t position x
        grad_hamiltonian = (
            grad_nll(x)
            - 0.5 * jnp.einsum("ijk,j,k->i", jac_metric, w, w)
            + 0.5 * jac_det_metric
        )
        return hamiltonian, grad_hamiltonian

    cholesky_metric, det_metric, jac_metric, jac_det_metric = __riemann_metric(x_init)

    def fp_p(_, val):
        """Fixed point iteration step for momentum update"""
        (
            _,
            x,
            p,
            grad_hamiltonian,
            cholesky_metric,
            det_metric,
            jac_metric,
            jac_det_metric,
        ) = val
        p_new = p - 0.5 * eps * grad_hamiltonian
        _, grad_hamiltonian = __hamiltonian(
            x, p_new, cholesky_metric, det_metric, jac_metric, jac_det_metric
        )
        return (
            p_new,
            x,
            p,
            grad_hamiltonian,
            cholesky_metric,
            det_metric,
            jac_metric,
            jac_det_metric,
        )

    def fp_x(_, val):
        """Fixed point iteration step for position update"""
        _, x, p, w, w_new, _, _, _, _ = val
        x_new = x + 0.5 * eps * (w + w_new)
        cholesky_metric, det_metric, jac_metric, jac_det_metric = __riemann_metric(
            x_new
        )
        w_new = cholesky_solve(cholesky_metric, p)
        return (
            x_new,
            x,
            p,
            w,
            w_new,
            cholesky_metric,
            det_metric,
            jac_metric,
            jac_det_metric,
        )

    def leapfrog(carry, _):
        """Generalized leapfrog integration step"""
        (
            x,
            p,
            _,
            grad_hamiltonian,
            cholesky_metric,
            det_metric,
            jac_metric,
            jac_det_metric,
        ) = carry

        # Update momentum implicitly (half step)
        fp_out_p = jax.lax.fori_loop(
            0,
            f_max,
            fp_p,
            (
                p,
                x,
                p,
                grad_hamiltonian,
                cholesky_metric,
                det_metric,
                jac_metric,
                jac_det_metric,
            ),
        )
        p = fp_out_p[0]
        w = cholesky_solve(cholesky_metric, p)

        # Update position implicitly (full step)
        fp_out_x = jax.lax.fori_loop(
            0,
            f_max,
            fp_x,
            (
                x,
                x,
                p,
                w,
                w,
                cholesky_metric,
                det_metric,
                jac_metric,
                jac_det_metric,
            ),
        )

        (
            x,
            _,
            _,
            _,
            _,
            cholesky_metric,
            det_metric,
            jac_metric,
            jac_det_metric,
        ) = fp_out_x

        # Final momentum explicit update (half step)
        _, grad_hamiltonian = __hamiltonian(
            x, p, cholesky_metric, det_metric, jac_metric, jac_det_metric
        )
        p = p - 0.5 * eps * grad_hamiltonian
        hamiltonian, grad_hamiltonian = __hamiltonian(
            x, p, cholesky_metric, det_metric, jac_metric, jac_det_metric
        )

        return (
            x,
            p,
            hamiltonian,
            grad_hamiltonian,
            cholesky_metric,
            det_metric,
            jac_metric,
            jac_det_metric,
        ), x

    def _loop(carry, _):
        (
            key,
            rej,
            x,
            cholesky_metric,
            det_metric,
            jac_metric,
            jac_det_metric,
        ) = carry
        key, subkey0, subkey1 = jax.random.split(key, 3)

        # Sample initial momentum from N(0, G(x))
        p = jax.random.normal(subkey0, x.shape)
        p = cholesky_metric @ p

        # Compute initial Hamiltonian
        hamiltonian, grad_hamiltonian = __hamiltonian(
            x, p, cholesky_metric, det_metric, jac_metric, jac_det_metric
        )

        jax.debug.print("{h}", h=hamiltonian)

        # Run generalized leapfrog integrator
        leap_carry, leap = jax.lax.scan(
            leapfrog,
            (
                x,
                p,
                hamiltonian,
                grad_hamiltonian,
                cholesky_metric,
                det_metric,
                jac_metric,
                jac_det_metric,
            ),
            None,
            t_max,
        )

        (
            x_new,
            _,
            new_hamiltonian,
            _,
            new_cholesky_metric,
            new_det_metric,
            new_jac_metric,
            new_jac_det_metric,
        ) = leap_carry

        # Accept-reject step
        u = jax.random.uniform(subkey1)
        condition = u < jnp.exp(hamiltonian - new_hamiltonian)

        new_rej = jax.lax.select(condition, rej, rej + 1)
        new_x = jax.lax.select(condition, x_new, x)
        new_cholesky_metric = jax.lax.select(
            condition, new_cholesky_metric, cholesky_metric
        )
        new_det_metric = jax.lax.select(condition, new_det_metric, det_metric)
        new_jac_metric = jax.lax.select(condition, new_jac_metric, jac_metric)
        new_jac_det_metric = jax.lax.select(
            condition, new_jac_det_metric, jac_det_metric
        )

        output = leap if return_path else new_x
        return (
            key,
            new_rej,
            new_x,
            new_cholesky_metric,
            new_det_metric,
            new_jac_metric,
            new_jac_det_metric,
        ), output

    carry, samples = jax.lax.scan(
        _loop,
        (
            key,
            0,
            x_init,
            cholesky_metric,
            det_metric,
            jac_metric,
            jac_det_metric,
        ),
        None,
        n_iter,
    )

    _, rej, _, _, _, _, _ = carry
    rejection_rate = rej / n_iter

    if return_path:
        samples = samples.reshape(-1, x_init.shape[-1])

    return rejection_rate, samples
