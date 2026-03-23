import jax
import jax.numpy as jnp

from typing import Callable, Tuple

def __riemann_metric(x, hE, dG):
    Gx = hE(x)
    GL = jnp.linalg.cholesky(Gx)
    Gdet = jnp.square(jnp.prod(jnp.diag(GL)))
    Gjac = dG(x)
    GidG = jnp.linalg.solve(GL, Gjac)
    GidG = jnp.linalg.solve(GL.T, GidG)
    return Gx, GL, Gdet, Gjac, GidG

def __hamiltonian(x, p, E, gE, GL, Gdet, Gjac, GidG):
    w = jnp.linalg.solve(GL, p)
    w = jnp.linalg.solve(GL.T, w)
    K = 0.5 * p.T @ w + 0.5 * jnp.log(Gdet)
    H = K + E(x)
    dH = (
        gE(x)
        + 0.5 * jnp.einsum('ijk,j,k->i', Gjac, w, w)
        - 0.5 * jnp.einsum('ijj', GidG)
    )
    return H, dH

def sample(
    key: jax.Array,
    prob: Callable[[jax.Array], float],
    x_init: jax.Array,
    n_iter: int,
    eps: float,
    t_max: int,
    f_max: int
) -> Tuple[float, jax.Array]:

    E  = lambda x: -jnp.log(prob(x))
    gE = jax.grad(E)
    hE = jax.hessian(E)
    dG = jax.jacfwd(hE)
    Gx, GL, Gdet, Gjac, GidG = __riemann_metric(x_init, hE, dG)

    def fp_p(carry, _):
        x, p, dH, Gx, GL, Gdet, Gjac, GidG = carry
        p_new = p - 0.5 * eps * dH
        _, dH = __hamiltonian(x, p_new, E, gE, GL, Gdet, Gjac, GidG)
        return (x, p, dH, Gx, GL, Gdet, Gjac, GidG), p_new

    def fp_x(carry, _):
        x, p, w, w_new, _, _, _, _, _ = carry
        x_new = x + 0.5 * eps * (w + w_new)
        Gx, GL, Gdet, Gjac, GidG = __riemann_metric(x_new, hE, dG)
        w_new = jnp.linalg.solve(GL, p)
        w_new = jnp.linalg.solve(GL.T, w_new)
        return (x, p, w, w_new, Gx, GL, Gdet, Gjac, GidG), x_new

    def leapfrog(carry, _):
        x, p, _, dH, Gx, GL, Gdet, Gjac, GidG = carry

        _, fp_arr_p = jax.lax.scan(
            fp_p,
            (x, p, dH, Gx, GL, Gdet, Gjac, GidG),
            None,
            f_max,
        )

        p = fp_arr_p[-1]
        w = jnp.linalg.solve(GL, p)
        w = jnp.linalg.solve(GL.T, w)

        carry, fp_arr_x = jax.lax.scan(
            fp_x,
            (x, p, w, w, Gx, GL, Gdet, Gjac, GidG),
            None,
            f_max,
        )

        _, _, _, _, Gx, GL, Gdet, Gjac, GidG = carry
        x = fp_arr_x[-1]
        _, dH = __hamiltonian(x, p, E, gE, GL, Gdet, Gjac, GidG)
        p = p - 0.5 * eps * dH
        H, dH = __hamiltonian(x, p, E, gE, GL, Gdet, Gjac, GidG)

        return (x, p, H, dH, Gx, GL, Gdet, Gjac, GidG), x

    def _loop(carry, _):
        key, rej, x, Gx, GL, Gdet, Gjac, GidG = carry
        key, subkey0, subkey1 = jax.random.split(key, 3)

        p = jax.random.normal(subkey0, x.shape)
        p = GL @ p

        H, dH = __hamiltonian(x, p, E, gE, GL, Gdet, Gjac, GidG)

        leap_carry, leap = jax.lax.scan(
            leapfrog,
            (x, p, H, dH, Gx, GL, Gdet, Gjac, GidG),
            None,
            t_max,
        )

        x_new, _, H_new, _, Gx_new, GL_new, Gdet_new, Gjac_new, GidG_new = leap_carry

        u = jax.random.uniform(subkey1)
        condition = u < jnp.exp(H - H_new)

        new_rej = jax.lax.select(condition, rej, rej+1)
        new_x = jax.lax.select(condition, x_new, x)
        new_Gx = jax.lax.select(condition, Gx_new, Gx)
        new_GL = jax.lax.select(condition, GL_new, GL)
        new_Gdet = jax.lax.select(condition, Gdet_new, Gdet)
        new_Gjac = jax.lax.select(condition, Gjac_new, Gjac)
        new_GidG = jax.lax.select(condition, GidG_new, GidG)

        return (key, new_rej, new_x, new_Gx, new_GL, new_Gdet, new_Gjac, new_GidG), new_x

    carry, samples = jax.lax.scan(
        _loop,
        (key, 0, x_init, Gx, GL, Gdet, Gjac, GidG),
        None,
        n_iter,
    )

    _, rej, _, _, _, _, _, _ = carry
    rejection_rate = rej / n_iter

    return rejection_rate, samples



def sample_with_path(
    key: jax.Array,
    prob: Callable[[jax.Array], float],
    x_init: jax.Array,
    n_iter: int,
    eps: float,
    t_max: int,
    f_max: int
) -> Tuple[float, jax.Array]:

    E  = lambda x: -jnp.log(prob(x))
    gE = jax.grad(E)
    hE = jax.hessian(E)
    dG = jax.jacfwd(hE)
    Gx, GL, Gdet, Gjac, GidG = __riemann_metric(x_init, hE, dG)

    def fp_p(carry, _):
        x, p, dH, Gx, GL, Gdet, Gjac, GidG = carry
        p_new = p - 0.5 * eps * dH
        _, dH = __hamiltonian(x, p_new, E, gE, GL, Gdet, Gjac, GidG)
        return (x, p, dH, Gx, GL, Gdet, Gjac, GidG), p_new

    def fp_x(carry, _):
        x, p, w, w_new, _, _, _, _, _ = carry
        x_new = x + 0.5 * eps * (w + w_new)
        Gx, GL, Gdet, Gjac, GidG = __riemann_metric(x_new, hE, dG)
        w_new = jnp.linalg.solve(GL, p)
        w_new = jnp.linalg.solve(GL.T, w_new)
        return (x, p, w, w_new, Gx, GL, Gdet, Gjac, GidG), x_new

    def leapfrog(carry, _):
        x, p, _, dH, Gx, GL, Gdet, Gjac, GidG = carry

        _, fp_arr_p = jax.lax.scan(
            fp_p,
            (x, p, dH, Gx, GL, Gdet, Gjac, GidG),
            None,
            f_max,
        )

        p = fp_arr_p[-1]
        w = jnp.linalg.solve(GL, p)
        w = jnp.linalg.solve(GL.T, w)

        carry, fp_arr_x = jax.lax.scan(
            fp_x,
            (x, p, w, w, Gx, GL, Gdet, Gjac, GidG),
            None,
            f_max,
        )

        _, _, _, _, Gx, GL, Gdet, Gjac, GidG = carry
        x = fp_arr_x[-1]
        _, dH = __hamiltonian(x, p, E, gE, GL, Gdet, Gjac, GidG)
        p = p - 0.5 * eps * dH
        H, dH = __hamiltonian(x, p, E, gE, GL, Gdet, Gjac, GidG)

        return (x, p, H, dH, Gx, GL, Gdet, Gjac, GidG), x

    def _loop(carry, _):
        key, rej, x, Gx, GL, Gdet, Gjac, GidG = carry
        key, subkey0, subkey1 = jax.random.split(key, 3)

        p = jax.random.normal(subkey0, x.shape)
        p = GL @ p

        H, dH = __hamiltonian(x, p, E, gE, GL, Gdet, Gjac, GidG)

        leap_carry, leap = jax.lax.scan(
            leapfrog,
            (x, p, H, dH, Gx, GL, Gdet, Gjac, GidG),
            None,
            t_max,
        )

        x_new, _, H_new, _, Gx_new, GL_new, Gdet_new, Gjac_new, GidG_new = leap_carry

        u = jax.random.uniform(subkey1)
        condition = u < jnp.exp(H - H_new)

        new_rej = jax.lax.select(condition, rej, rej+1)
        new_x = jax.lax.select(condition, x_new, x)
        new_Gx = jax.lax.select(condition, Gx_new, Gx)
        new_GL = jax.lax.select(condition, GL_new, GL)
        new_Gdet = jax.lax.select(condition, Gdet_new, Gdet)
        new_Gjac = jax.lax.select(condition, Gjac_new, Gjac)
        new_GidG = jax.lax.select(condition, GidG_new, GidG)

        return (key, new_rej, new_x, new_Gx, new_GL, new_Gdet, new_Gjac, new_GidG), leap

    carry, path = jax.lax.scan(
        _loop,
        (key, 0, x_init, Gx, GL, Gdet, Gjac, GidG),
        None,
        n_iter,
    )

    _, rej, _, _, _, _, _, _ = carry
    rejection_rate = rej / n_iter

    path = path.reshape(-1, x_init.shape[-1])

    return rejection_rate, path
