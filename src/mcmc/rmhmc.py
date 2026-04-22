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
    def neg_log_prob(x):
        """Negative log-probability (potential energy)"""
        return -log_prob(x)

    # Compute gradient, and Hessian of the negative log-probability
    grad_nll = jax.grad(neg_log_prob)
    hessian_nll = jax.hessian(neg_log_prob)
    # Jacobain of the Hessian gives the derivatives of the metric tensor
    jac_hessian_nll = jax.jacfwd(hessian_nll)

    @jax.jit
    def __riemann_metric(x):
        # Compute the Riemannian metric tensor G(x) (using the Hessian)
        metric = hessian_nll(x)
        # Cholesky decomposition of G(x)
        cholesky_metric = jnp.linalg.cholesky(metric)
        # Determinant of G(x)
        det_metric = jnp.square(jnp.prod(jnp.diag(cholesky_metric)))
        # Jacobian of G(x)
        jac_metric = jac_hessian_nll(x)
        # Compute G(x)^{-1} dG(x)
        metric_inv_dot_jac = jnp.linalg.solve(cholesky_metric, jac_metric)
        metric_inv_dot_jac = jnp.linalg.solve(cholesky_metric.T, metric_inv_dot_jac)
        return metric, cholesky_metric, det_metric, jac_metric, metric_inv_dot_jac

    @jax.jit
    def __hamiltonian(
        x, p, cholesky_metric, det_metric, jac_metric, metric_inv_dot_jac
    ):
        # Compute kinetic energy K = 1/2 p^T G(x)^{-1} p + 1/2 log|G(x)|
        w = jnp.linalg.solve(cholesky_metric, p)
        w = jnp.linalg.solve(cholesky_metric.T, w)
        K = 0.5 * p.T @ w + 0.5 * jnp.log(det_metric)
        # Total Hamiltonian H = K + U
        hamiltonian = K + neg_log_prob(x)
        # Partial derivative of the Hamiltonian w.r.t position x
        grad_hamiltonian = (
            grad_nll(x)
            + 0.5 * jnp.einsum("ijk,j,k->i", jac_metric, w, w)
            - 0.5 * jnp.einsum("ijj", metric_inv_dot_jac)
        )
        return hamiltonian, grad_hamiltonian

    metric, cholesky_metric, det_metric, jac_metric, metric_inv_dot_jac = (
        __riemann_metric(x_init)
    )

    def fp_p(carry, _):
        """Fixed point iteration step for momentum update"""
        (
            x,
            p,
            grad_hamiltonian,
            metric,
            cholesky_metric,
            det_metric,
            jac_metric,
            metric_inv_dot_jac,
        ) = carry
        p_new = p - 0.5 * eps * grad_hamiltonian
        _, grad_hamiltonian = __hamiltonian(
            x, p_new, cholesky_metric, det_metric, jac_metric, metric_inv_dot_jac
        )
        return (
            x,
            p,
            grad_hamiltonian,
            metric,
            cholesky_metric,
            det_metric,
            jac_metric,
            metric_inv_dot_jac,
        ), p_new

    def fp_x(carry, _):
        """Fixed point iteration step for position update"""
        x, p, w, w_new, _, _, _, _, _ = carry
        x_new = x + 0.5 * eps * (w + w_new)
        metric, cholesky_metric, det_metric, jac_metric, metric_inv_dot_jac = (
            __riemann_metric(x_new)
        )
        w_new = jnp.linalg.solve(cholesky_metric, p)
        w_new = jnp.linalg.solve(cholesky_metric.T, w_new)
        return (
            x,
            p,
            w,
            w_new,
            metric,
            cholesky_metric,
            det_metric,
            jac_metric,
            metric_inv_dot_jac,
        ), x_new

    def leapfrog(carry, _):
        """Generalized leapfrog integration step"""
        (
            x,
            p,
            _,
            grad_hamiltonian,
            metric,
            cholesky_metric,
            det_metric,
            jac_metric,
            metric_inv_dot_jac,
        ) = carry

        # Update momentum implicitly (half step)
        _, fp_arr_p = jax.lax.scan(
            fp_p,
            (
                x,
                p,
                grad_hamiltonian,
                metric,
                cholesky_metric,
                det_metric,
                jac_metric,
                metric_inv_dot_jac,
            ),
            None,
            f_max,
        )

        p = fp_arr_p[-1]
        w = jnp.linalg.solve(cholesky_metric, p)
        w = jnp.linalg.solve(cholesky_metric.T, w)

        # Update position implicitly (full step)
        carry, fp_arr_x = jax.lax.scan(
            fp_x,
            (
                x,
                p,
                w,
                w,
                metric,
                cholesky_metric,
                det_metric,
                jac_metric,
                metric_inv_dot_jac,
            ),
            None,
            f_max,
        )

        (
            _,
            _,
            _,
            _,
            metric,
            cholesky_metric,
            det_metric,
            jac_metric,
            metric_inv_dot_jac,
        ) = carry
        x = fp_arr_x[-1]

        # Final momentum explicit update (half step)
        _, grad_hamiltonian = __hamiltonian(
            x, p, cholesky_metric, det_metric, jac_metric, metric_inv_dot_jac
        )
        p = p - 0.5 * eps * grad_hamiltonian
        hamiltonian, grad_hamiltonian = __hamiltonian(
            x, p, cholesky_metric, det_metric, jac_metric, metric_inv_dot_jac
        )

        return (
            x,
            p,
            hamiltonian,
            grad_hamiltonian,
            metric,
            cholesky_metric,
            det_metric,
            jac_metric,
            metric_inv_dot_jac,
        ), x

    def _loop(carry, _):
        (
            key,
            rej,
            x,
            metric,
            cholesky_metric,
            det_metric,
            jac_metric,
            metric_inv_dot_jac,
        ) = carry
        key, subkey0, subkey1 = jax.random.split(key, 3)

        # Sample initial momentum from N(0, G(x))
        p = jax.random.normal(subkey0, x.shape)
        p = cholesky_metric @ p

        # Compute initial Hamiltonian
        hamiltonian, grad_hamiltonian = __hamiltonian(
            x, p, cholesky_metric, det_metric, jac_metric, metric_inv_dot_jac
        )

        # Run generalized leapfrog integrator
        leap_carry, leap = jax.lax.scan(
            leapfrog,
            (
                x,
                p,
                hamiltonian,
                grad_hamiltonian,
                metric,
                cholesky_metric,
                det_metric,
                jac_metric,
                metric_inv_dot_jac,
            ),
            None,
            t_max,
        )

        (
            x_new,
            _,
            new_hamiltonian,
            _,
            new_metric,
            new_cholesky_metric,
            new_det_metric,
            new_jac_metric,
            new_metric_inv_dot_jac,
        ) = leap_carry

        # Accept-reject step
        u = jax.random.uniform(subkey1)
        condition = u < jnp.exp(hamiltonian - new_hamiltonian)

        new_rej = jax.lax.select(condition, rej, rej + 1)
        new_x = jax.lax.select(condition, x_new, x)
        new_metric = jax.lax.select(condition, new_metric, metric)
        new_cholesky_metric = jax.lax.select(
            condition, new_cholesky_metric, cholesky_metric
        )
        new_det_metric = jax.lax.select(condition, new_det_metric, det_metric)
        new_jac_metric = jax.lax.select(condition, new_jac_metric, jac_metric)
        new_metric_inv_dot_jac = jax.lax.select(
            condition, new_metric_inv_dot_jac, metric_inv_dot_jac
        )

        output = leap if return_path else new_x
        return (
            key,
            new_rej,
            new_x,
            new_metric,
            new_cholesky_metric,
            new_det_metric,
            new_jac_metric,
            new_metric_inv_dot_jac,
        ), output

    carry, samples = jax.lax.scan(
        _loop,
        (
            key,
            0,
            x_init,
            metric,
            cholesky_metric,
            det_metric,
            jac_metric,
            metric_inv_dot_jac,
        ),
        None,
        n_iter,
    )

    _, rej, _, _, _, _, _, _ = carry
    rejection_rate = rej / n_iter

    if return_path:
        samples = samples.reshape(-1, x_init.shape[-1])

    return rejection_rate, samples
