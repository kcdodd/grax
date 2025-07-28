from jax import (
  numpy as jnp)
import jax
from functools import partial
import numpy as np

#===============================================================================
def uv(xhat, xscale):
  _x = xhat[0]*(1/xscale[0])
  _t = xhat[1]*(1/xscale[1])

  u = 0.5 + _t - _x
  v = 0.5 + _t + _x
  return jnp.stack([u, v])

#===============================================================================
def weights(xhat, xscale):
  u, v = uv(xhat, xscale)
  return (1-u)*(1-v), u*(1-v), (1-u)*v, u*v

#===============================================================================
def coord(xhat, X, scale):
  shape = xhat.shape
  xhat = jnp.atleast_2d(xhat)
  xhat = xhat.reshape(shape[0], -1)[:,None]
  w00, w01, w10, w11 = weights(xhat, scale)
  # print(f"{w00.shape=}")
  x = w00*X[0,0,:,None] + w01*X[0,1,:,None] + w10*X[1,0,:,None] + w11*X[1,1,:,None]
  # print(f"{x.shape=}")
  return x.reshape(*shape)

#===============================================================================
def vector_pick(*args, vec_fn, dim):
  x = vec_fn(jnp.stack(args, axis=0))[dim]
  # print(f"{dim=}, {x.shape=}")
  return x

#===============================================================================
def tensor_pick(*args, tensor_fn, dims):
  x = tensor_fn(jnp.stack(args, axis=0))[dims]
  # print(f"{dim=}, {x.shape=}")
  return x

#===============================================================================
def coord_jac(
    xhat,
    x_fn):

  shape = xhat.shape[1:]
  xhat = xhat.reshape(xhat.shape[0], -1)
  n = len(xhat)

  J = jnp.stack([
    jnp.stack([
        jax.vmap(jax.grad(partial(vector_pick, vec_fn=x_fn, dim=i), argnums=j))(*xhat)
        for j in range(n)],
      axis=0)
    for i in range(n)],
    axis=0)

  return J.reshape(2, 2, *shape)

#===============================================================================
def metric_inv(
    xhat,
    x_fn,
    sig):

  shape = xhat.shape[1:]
  xhat = xhat.reshape(xhat.shape[0], -1)
  J = coord_jac(xhat, x_fn)
  ginv = jnp.einsum('ij...,j,kj...->ik...', J, sig, J)
  return ginv.reshape(2, 2, *shape)

#===============================================================================
def metric(
    xhat,
    x_fn,
    sig):

  shape = xhat.shape[1:]
  xhat = xhat.reshape(xhat.shape[0], -1)
  ginv = metric_inv(xhat, x_fn, sig)
  g = jnp.linalg.inv(ginv.transpose(2,0,1)).transpose(1, 2, 0)
  return g.reshape(2, 2, *shape)

#===============================================================================
def connection(
    xhat,
    x_fn,
    sig):
  ...


#===============================================================================
xhat = jnp.stack(jnp.meshgrid(
  jnp.linspace(-0.5, 0.5, 1024),
  jnp.linspace(-0.5, 0.5, 1024),
  indexing='ij'),
  axis=0)

sig = jnp.array([1.0, -1.0])

delta = 0.1
X = jnp.stack([
  jnp.array([
    [0.0, -0.5],
    [0.5, 0.0]]),
  jnp.array([
    [-0.5, 0.0-delta],
    [0.0-delta, 0.5]])],
  axis=-1)

x = coord(xhat, X, [1.0, 1.0])

g = metric(xhat, partial(coord, X=X, scale=[1.0, 1.0]), sig)



