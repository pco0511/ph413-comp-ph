from functools import partial

import jax
import jax.numpy as jnp

from jaxtyping import Array, Int, Float, PRNGKeyArray

import tqdm

import matplotlib.pyplot as plt



@partial(
    jax.jit, 
    static_argnames=("width", "height", "margin", 
                     "min_rad", "max_rad", "min_area",
                     "max_area", "continue_prob")
)
def create_random_mask(
    key: PRNGKeyArray,
    width: int,
    height: int,
    margin: int,
    min_rad: int,
    max_rad: int,
    min_area: int,
    max_area: int,
    continue_prob: float
):
    """
    Generate a random connected-circle mask.
    key            : PRNGKey
    width, height : output height/width
    margin         : min dist from edge
    rmin, rmax     : circle radius range
    amin, amax     : min/max mask area
    continue_prob  : continue probability once amin is reached
    """
    i = jnp.arange(height)[:, None]
    j = jnp.arange(width)[None, :]
    r = (i - height / 2)**2 + (j - width / 2)**2
    big_circle = r < (height / 2 - margin)**2
    
    # state = (mask, area, keep_going, key)
    init = (jnp.zeros((height, width), jnp.uint8), 0, True, key)
    
    def cond(state):
        return state[2]
    
    def body(state):
        mask, area, _, key = state
        key, key0, key1, key2, key3 = jax.random.split(key, 5)

        # sample circle
        ci = jax.random.randint(key0, (), 0, height)
        cj = jax.random.randint(key1, (), 0, width)
        cr = jax.random.randint(key2, (), min_rad, max_rad)
        circle = (i - ci)**2 + (j - cj)**2 < cr**2

        # reject if outside big circle
        bad = jnp.any(circle & (~big_circle))

        # require connectivity if non-empty
        conn_ok = (area == 0) | (mask[ci, cj] == 1)

        # decide add or skip
        add = (~bad) & conn_ok
        new_mask = jnp.where(add, mask | circle, mask)

        # update area & continue flag
        new_area = jnp.sum(new_mask)
        cont = jnp.where(
            new_area > min_area,
            (jax.random.uniform(key3) < continue_prob) & (new_area < max_area),
            True
        )

        return (new_mask, new_area, cont, key)
    
    mask_final, _, _, _ = jax.lax.while_loop(cond, body, init)
    return mask_final