"""Detect mathematical landmarks."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

__all__ = [
    "landmarks_type2",
]


def landmarks_type2(x, Ys, Ls, sigma):
    """Mathematical landmarks for heavy edge profiles without trough.

    The following landmarks are detected:
    1. Contact point.
    2. Peak point.
    3. Knee point between plateau and peak.

    Parameters
    ----------
    x : array of shape (M,)
        X grid of profiles.
    Ys : array of shape (N, M)
        Height data of N profiles.
    Ls : array of shape (N,) and dtype=int
        Length of each profile.
    sigma : scalar
        Standard deviation of Gaussian filter for smoothing.

    Returns
    -------
    array of shape (N, 2, 3)
        X and Y coordinates of landmarks.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_landmarks import landmarks_type2
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> lm = landmarks_type2(x, Ys, Ls, 32)
    >>> lm.shape
    (22, 2, 3)
    """
    ret = []
    for Y, L in zip(Ys, Ls):
        idxs = _landmarks_type2(Y[:L], sigma)
        ret.append([x[idxs], Y[idxs]])
    return np.array(ret)


def _landmarks_type2(Y, sigma):
    cp = len(Y) - 1

    Y_smooth = gaussian_filter1d(Y, sigma)
    peaks, _ = find_peaks(Y_smooth)
    peak = peaks[-1]

    Y_ = Y_smooth[:peak]
    pts = np.column_stack([np.arange(len(Y_)), Y_])
    x, y = pts - pts[0], pts[-1] - pts[0]
    dists = x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    slope = np.diff(dists)
    (extrema,) = np.nonzero(np.diff(np.sign(slope)))
    K_pos = extrema[slope[extrema] > 0]
    knee = K_pos[np.argmax(np.abs(dists[K_pos]))]

    return np.array([cp, peak, knee])
