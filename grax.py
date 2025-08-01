from typing_extensions import (
  TypeVar)
from jax import (
  numpy as jnp)
import jax
from pathlib import Path
from jax.experimental import ode
import numpy as np
from nohm.core import (
  Enclose,
  Field,
  Numeric,
  Static,
  Floating,
  getLogger)
from nohm.solve import (
  step_adapt)
from nohm.plot import (
  Figure,
  Plot2D,
  Plot,
  Plot1D,
  Line)

log = getLogger(__name__)
PLOT_DIR = Path(__file__).parent/f"plots_{Path(__file__).name.rstrip('.py')}"
PLOT_DIR.mkdir(exist_ok=True)

D = TypeVar('D')

#===============================================================================
def metric_inv_point(metric_fn, coord: Floating[D]) -> Floating[D,D]:
  return jnp.linalg.inv(metric_fn(coord))

#===============================================================================
def connection_point(metric_fn, coord: Floating[D]) -> Floating[D,D,D]:
    # [D] -> [D,D]
    ginv = metric_inv_point(metric_fn, coord)

    # [D,D] -> [D,D,D]
    # g_{ij,k}
    dg = jax.jacrev(metric_fn)(coord)
    print(f"{dg.shape=}")
    print(dg)

    gamma = 0.5*(
      jnp.einsum('im,mjk->ijk', ginv, dg)
      + jnp.einsum('im,mkj->ijk', ginv, dg)
      - jnp.einsum('im,jkm->ijk', ginv, dg))

    return gamma

#===============================================================================
def ricci_scalar_point(metric_fn, coord: Floating[D]) -> Floating[D,D]:
  ginv = metric_inv_point(metric_fn, coord)
  gamma = connection_point(metric_fn, coord)

  # Γ^{i}_{jk,l}
  dgamma = jax.jacrev(Enclose(connection_point, metric_fn))(coord)

  return (
    jnp.einsum('ij,kijk', ginv, dgamma)
    - jnp.einsum('ij,kikj', ginv, dgamma)
    + jnp.einsum('ij,mij,kkm', ginv, gamma, gamma)
    - jnp.einsum('ij,mik,kjm', ginv, gamma, gamma))

#===============================================================================
def lagrangian_density(metric_fn, coord: Floating[D]) -> Floating:
  g = metric_fn(coord)
  R = ricci_scalar_point(metric_fn, coord)
  return jnp.linalg.det(g)*R

#===============================================================================
def minkowski(coord, sign: float = 1.0):
  d = len(coord)

  return sign*jnp.block([
    [jnp.array([[1.0]]), jnp.zeros((1,d-1))],
    [jnp.zeros((d-1,1)), -jnp.eye(d-1)]])

#===============================================================================
def schwarzschild(coord, rg = 1.0, rs = 0.25):
  rg = jnp.maximum(rg, rs)
  d = len(coord)
  t, r, theta, phi = coord

  r_exterior = jnp.where(r > rg, r, rg)
  r_interior = jnp.where(r < rg, r, rg)

  rsq_interior = r_interior**2
  rsq_exterior = r_exterior**2

  a = jnp.minimum(1.0, rsq_interior*rs/rg**3)
  b = jnp.minimum(1.0, rs/r_exterior)

  interior = jnp.array([
    -0.25*(3*(1-rs/rg)**0.5 - (1-a)**0.5)**2,
    1/(1-a),
    rsq_interior*jnp.cos(phi)**2,
    rsq_interior])

  exterior = jnp.array([
    -(1-b),
    1/(1-b),
    rsq_exterior*jnp.cos(phi)**2,
    rsq_exterior])

  print(f"{interior.dtype=}")

  return jnp.diag(jnp.where(r < rg, interior, exterior))

#===============================================================================
# coord = jnp.array([0.0, 0.0])
r = jnp.linspace(0, 4, 1000)
coord = jnp.stack([
  jnp.zeros_like(r),
  r,
  jnp.zeros_like(r),
  jnp.zeros_like(r)], axis=1)

print(f"{coord.shape=}")

# metric_fn = minkowski
metric_fn = schwarzschild

g = jax.vmap(metric_fn)(coord)
print(f"{g.shape=}")
ginv = jnp.linalg.inv(g)

# Figure(Plot1D([Line(g[:,0,0])]), Plot1D([Line(g[:,1,1])]), Plot1D([Line(ginv[:,0,0])]), Plot1D([Line(ginv[:,1,1])])).fig()

# gamma = jax.vmap(connection_point)(metric_fn, coord)

R = jax.vmap(ricci_scalar_point)(metric_fn, coord)
L = jax.vmap(lagrangian_density)(metric_fn, coord)

extent=[[r[0], r[-1]], [jnp.amin(g),jnp.amax(g)]]

Figure(
  Plot1D([Line(g[:,0,0])]),
  Plot1D([Line(g[:,1,1])]),
  Plot1D([Line(R)]),
  Plot1D([Line(L)])).fig()
