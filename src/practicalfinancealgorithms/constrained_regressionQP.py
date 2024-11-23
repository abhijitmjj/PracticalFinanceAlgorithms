import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP


def constrained_regression(X, y, G, h, A, b, alpha):
    # Ensure inputs are JAX arrays
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n_features = X.shape[1]

    # Formulate the QP problem
    P = X.T @ X + alpha * jnp.eye(n_features)
    q = -X.T @ y

    # Inequality constraints: Gβ ≤ h
    if G is not None and h is not None:
        G = jnp.asarray(G)
        h = jnp.asarray(h)
        l_ineq = -jnp.inf * jnp.ones_like(h)
        u_ineq = h
    else:
        G = jnp.empty((0, n_features))
        l_ineq = jnp.empty((0,))
        u_ineq = jnp.empty((0,))

    # Equality constraints: Aβ = b
    if A is not None and b is not None:
        A = jnp.asarray(A)
        b = jnp.asarray(b)
        l_eq = b
        u_eq = b
    else:
        A = jnp.empty((0, n_features))
        l_eq = jnp.empty((0,))
        u_eq = jnp.empty((0,))

    # Combine constraints
    A_combined = jnp.vstack([G, A])
    l_combined = jnp.concatenate([l_ineq, l_eq])
    u_combined = jnp.concatenate([u_ineq, u_eq])

    # Define matvec function for Q
    def matvec_Q(_, x):
        return P @ x

    # Initialize the solver
    osqp = BoxOSQP(matvec_Q=matvec_Q, tol=1e-5, maxiter=1000, verbose=0)

    # Run the solver
    solution = osqp.run(
        init_params=None,
        params_obj=(None, q),
        params_eq=A_combined,
        params_ineq=(l_combined, u_combined),
    )

    # Extract the solution
    beta_opt = solution.params.primal[0]

    return beta_opt


# Example Usage
if __name__ == "__main__":
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
