"""Conversion between configuration matrix and pre-shape."""

import numpy as np
from scipy.linalg import helmert

__all__ = [
    "preshape",
    "dual_preshape",
]


def preshape(Xs):
    """Convert configuration matrices to pre-shapes.

    Conversion is done using the Helmert sub-matrix.

    Parameters
    ----------
    Xs : array, shape (N, m, k)
        `N` configuration matrices of `k` landmarks in dimension `m`.

    Returns
    -------
    Zs : array, shape (N, m, k-1)
        `N` pre-shape matrices.

    See Also
    --------
    dual_preshape
        Pre-shape in configuration matrix space.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_landmarks import pseudo_landmarks, preshape
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> Zs = preshape(pseudo_landmarks(x, Ys, Ls, 10))
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*Zs.transpose(1, 2, 0))
    """
    _, _, k = Xs.shape
    H = helmert(k)
    HX = np.inner(Xs, H)
    scale = np.linalg.norm(HX, axis=(1, 2), keepdims=True)
    Zs = HX / scale
    return Zs


def dual_preshape(Xs):
    """Pre-shape in configuration matrix space.

    Parameters
    ----------
    Xs : array, shape (N, m, k)
        `N` configuration matrices of `k` landmarks in dimension `m`.

    Returns
    -------
    Zs : array, shape (N, m, k)
        `N` pre-shape matrices.

    See Also
    --------
    preshape
        Pre-shape in its original space.

    Notes
    -----
    Because location and scale information is lost during the pre-shaping process,
    *Zs* is rank-deficient and has unit norm.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_landmarks import pseudo_landmarks, dual_preshape
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> Zs = dual_preshape(pseudo_landmarks(x, Ys, Ls, 10))
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*Zs.transpose(1, 2, 0))
    """
    _, _, k = Xs.shape
    H = helmert(k)
    HX = np.inner(Xs, H)
    scale = np.linalg.norm(HX, axis=(1, 2), keepdims=True)
    Zs = HX / scale
    hat = H.T @ np.linalg.inv(H @ H.T)
    return np.inner(Zs, hat)
