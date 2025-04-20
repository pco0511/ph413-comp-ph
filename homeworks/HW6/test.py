# YOUR CODE HERE (30pts)

# YOUR GENERATE TRAINING SET CODE HERE
import jax
from jax import numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

import tqdm

xrange = [0, 1]
trange = [0, 1]

def dataset_gen(n_samples=1000):
    x = jax.random.uniform(xrange[0], xrange[1], (n_samples, 1))