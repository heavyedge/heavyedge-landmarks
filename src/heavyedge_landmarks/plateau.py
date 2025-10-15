"""Detect plateau boundary."""

import numpy as np

__all__ = []


def _ols(Xi, Y):
    XT_X_inv = np.linalg.inv(Xi.T @ Xi)
    params = XT_X_inv @ (Xi.T @ Y)
    return params


def _segreg(x, Y, psi0, tol=1e-5, maxiter=30):
    r"""Segmented regression with one breakpoint.

    Parameters
    ----------
    x, Y : (M,) ndarray
        Data points.
    psi0 : scalar
        Initial guess for breakpoint coordinate.
    tol : float, default=1e-5
        Convergence tolerance.
    maxiter : int, default=30
        Force break after this iterations.

    Returns
    -------
    params : (4,) ndarray
        Estimated parameters: b0, b1, b2, psi.
    reached_max : bool
        Iteration is finished not by convergence but by reaching maximum iteration.
    """
    Xi = np.array(
        [
            np.ones_like(x),
            x,
            (x - psi0) * np.heaviside(x - psi0, 0),
            -np.heaviside(x - psi0, 0),
        ]
    ).T

    b0, b1, b2, gamma = _ols(Xi, Y)
    RSS = np.sum((Y - _segreg_predict(x, b0, b1, b2, psi0)) ** 2)

    psi_converged = False
    for _ in range(maxiter):
        RSS_new = RSS
        lamda = 1
        while True:
            psi0_new = psi0 + lamda * gamma / b2
            RSS_new = np.sum((Y - _segreg_predict(x, b0, b1, b2, psi0_new)) ** 2)
            lamda /= 2

            if (psi0_new <= x[0]) or (psi0_new >= x[-1]):
                # exceeded domain; make step size smaller
                continue
            if RSS_new >= RSS:
                # RSS not decreased; make step size smaller
                continue
            psi_converged = np.abs(psi0 - psi0_new) <= tol
            if psi_converged:
                break

        if not psi_converged:
            psi_converged = np.abs(psi0 - psi0_new) <= tol
        if psi_converged:
            psi0 = psi0_new
            reached_max = False
            break

        psi0 = psi0_new
        RSS = RSS_new
        Xi[:, 2] = (x - psi0) * np.heaviside(x - psi0, 0)
        Xi[:, 3] = -np.heaviside(x - psi0, 0)
        b0, b1, b2, gamma = _ols(Xi, Y)
    else:
        reached_max = True

    params = np.array([b0, b1, b2, psi0_new])
    return params, reached_max


def _segreg_predict(x, b0, b1, b2, psi):
    x = np.asarray(x)
    return b0 + b1 * x + b2 * (x - psi) * np.heaviside(x - psi, 0)
