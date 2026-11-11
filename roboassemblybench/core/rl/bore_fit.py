"""Balance the worst bore-section error, rather than their squared average."""
import numpy as np


def minimax_correction(jacobian, error, allowance):
    """Return a four-DOF line correction and its linearized worst error.

    Translation variables are millimetres, rotation variables are 0.02 rad.
    The small convex problem has an analytic constraint Jacobian. Compatible
    least-squares fits take the fast path without invoking the optimizer.
    """
    scale = np.diag([1.0, 1.0, 20.0, 20.0])
    matrix = np.asarray(jacobian) @ scale
    rhs = np.asarray(error) * 1000.0
    if np.linalg.matrix_rank(matrix) < 4:
        raise ValueError('axis_alignment needs distinct axial section heights')
    initial = np.linalg.lstsq(matrix, -rhs, rcond=None)[0]

    def residual(value):
        return (rhs + matrix @ value).reshape(-1, 2)

    def worst(value):
        return float(np.max(np.linalg.norm(residual(value), axis=1)))

    best, method = initial, 'least_squares'
    if worst(initial) > allowance * 1000.0 + 1e-8:
        from scipy.optimize import minimize

        blocks = matrix.reshape(-1, 2, 4)

        def constraints(value):
            return value[4] - np.sqrt(np.sum(residual(value[:4]) ** 2, axis=1) + 1e-24)

        def constraint_jacobian(value):
            r = residual(value[:4])
            norms = np.sqrt(np.sum(r * r, axis=1) + 1e-24)
            return np.column_stack((-np.einsum('ni,nij->nj', r / norms[:, None], blocks), np.ones(len(r))))

        result = minimize(
            lambda value: value[4],
            np.r_[initial, worst(initial)],
            jac=lambda value: np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            constraints=[dict(type='ineq', fun=constraints, jac=constraint_jacobian)],
            bounds=[(None, None)] * 4 + [(0.0, None)],
            method='SLSQP',
            options=dict(ftol=1e-10, maxiter=40),
        )
        if result.success and np.all(np.isfinite(result.x)) and worst(result.x[:4]) <= worst(initial) + 1e-7:
            best, method = result.x[:4], 'minimax'
        else:
            method = 'least_squares_fallback'
    return scale @ best * 0.001, worst(best) * 0.001, method
