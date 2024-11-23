from practicalfinancealgorithms.constrained_regressionQP import *

import jax.numpy as jnp

if __name__ == "__main__":
    # Load the data
    # Sample data
    X = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    y = jnp.array([7, 8, 9], dtype=jnp.float32)
    G = jnp.array([[1, 1], [-1, 0], [0, -1]], dtype=jnp.float32)
    h = jnp.array([10, 0, 0], dtype=jnp.float32)
    A = jnp.array([[1, 1]], dtype=jnp.float32)
    b = jnp.array([1], dtype=jnp.float32)
    alpha = 0.1

    # Solve the problem
    beta = constrained_regression(X, y, G, h, A, b, alpha)
    print("Optimal beta:", beta)