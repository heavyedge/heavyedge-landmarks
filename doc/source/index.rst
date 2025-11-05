.. HeavyEdge-Landmarks documentation master file, created by
   sphinx-quickstart on Wed Oct 15 09:22:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
HeavyEdge-Landmarks documentation
*********************************

.. module:: heavyedge_landmarks

.. plot:: plot-header.py
    :include-source: False

HeavyEdge-Landmarks is a Python package for locating landmarks from coating edge profiles and converting them to pre-shapes.

=========
Tutorials
=========

This section provides basic tutorials for beginners.

Preparing data
==============

Detecting landmarks requires profile and length data.
Here, we use preprocessed data distributed by the :mod:`heavyedge` package.

.. plot::
    :context: reset

    >>> from heavyedge import get_sample_path, ProfileData
    >>> with ProfileData(get_sample_path("Prep-Type1.h5")) as data:
    ...     x1 = data.x()
    ...     Ys1, Ls1, _ = data[:]
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x2 = data.x()
    ...     Ys2, Ls2, _ = data[:]
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     x3 = data.x()
    ...     Ys3, Ls3, _ = data[:]
    >>> import matplotlib.pyplot as plt
    ... plt.plot(x1, Ys1.T)
    ... plt.plot(x2, Ys2.T)
    ... plt.plot(x3, Ys3.T)

Locating landmarks
==================

Use :func:`pseudo_landmarks` to locate landmarks by equidistant sampling.
You need to specify the number of points `k` to sample.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import pseudo_landmarks
    >>> k = 10  # Number of landmarks
    >>> lm1 = pseudo_landmarks(x1, Ys1, Ls1, k)
    >>> import matplotlib.pyplot as plt
    ... plt.plot(x1, Ys1.T, color="gray", alpha=0.5)
    ... plt.plot(*lm1.transpose(1, 2, 0))

Use :func:`landmarks_type2` to locate feature points as landmarks, assuming a Type 2 shape which has a heavy edge peak but no trough.
You need to specify the standard deviation `sigma` of the Gaussian kernel for the function to internally smooth noise.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import landmarks_type2
    >>> sigma = 32  # Gaussian kernel std for noise smoothing
    >>> lm2 = landmarks_type2(x2, Ys2, Ls2, sigma)
    >>> plt.plot(x2, Ys2.T, color="gray", alpha=0.5)
    ... plt.plot(*lm2.transpose(1, 2, 0))

Use :func:`landmarks_type3` to locate feature points as landmarks, assuming a Type 3 shape which has a heavy edge peak and trough.
Like :func:`landmarks_type2`, you need to specify `sigma`.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import landmarks_type3
    >>> sigma = 32  # Gaussian kernel std for noise smoothing
    >>> lm3 = landmarks_type3(x3, Ys3, Ls3, sigma)
    >>> plt.plot(x3, Ys3.T, color="gray", alpha=0.5)
    ... plt.plot(*lm3.transpose(1, 2, 0))

Transforming to pre-shapes
==========================

In statistical shape analysis, a matrix of landmark coordinates from an object is called the *configuration matrix*.
Configuration matrices can be transformed to *pre-shapes*, where location and size information is removed, using :func:`preshape`.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import preshape
    >>> ps3 = preshape(lm3)
    >>> plt.plot(*ps3.transpose(1, 2, 0))

Pre-shapes exist in a different space from configuration matrices.
If you want to represent your pre-shape in the original space, use :func:`dual_preshape`.
Note that pre-shapes in the original space are rank-deficient.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import dual_preshape
    >>> dual_ps3 = dual_preshape(lm3)
    >>> plt.plot(*dual_ps3.transpose(1, 2, 0))

Fitting the plateau
===================

Plateaus detected by landmarks can be severely affected by noise or data artifacts.
For Type 2 profiles, :func:`plateau_type2` can be used for more robust plateau detection through nonlinear regression.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import plateau_type2
    >>> peaks, knees = lm2[:, 0, 1:].T
    >>> y0, m, xlast = plateau_type2(x2, Ys2, peaks, knees).T
    >>> x = np.stack([np.zeros(len(xlast)), xlast])
    >>> y = y0 + m * x
    >>> plt.plot(x2, Ys2.T, color="gray", alpha=0.5)
    ... plt.plot(x, y)

Likewise, :func:`plateau_type3` can be used for Type 3 profiles.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import plateau_type3
    >>> troughs, knees = lm3[:, 0, 2:].T
    >>> y0, m, xlast = plateau_type3(x3, Ys3, troughs, knees).T
    >>> x = np.stack([np.zeros(len(xlast)), xlast])
    >>> y = y0 + m * x
    >>> plt.plot(x3, Ys3.T, color="gray", alpha=0.5)
    ... plt.plot(x, y)


=============
How-to Guides
=============

This section provides guidance on how to use the library effectively.

Determining the sigma value
===========================

Choosing the right sigma value is crucial for effective noise smoothing.
A small sigma may not adequately smooth the data, while a large sigma can oversmooth and remove important features.
It's often useful to experiment with different sigma values and visually inspect the results.

A good choice for sigma is to use the value used for detecting contact points from raw data by the :mod:`heavyedge` package.
The code below shows that `sigma=32` properly detects contact points for all profiles.
Using this value for landmark detection will likely work well.

.. plot::
    :context: close-figs

    >>> from heavyedge import get_sample_path, RawProfileCsvs
    >>> from heavyedge.api import prep
    >>> raw = RawProfileCsvs(get_sample_path("Type3"))
    >>> sigma = 32
    >>> _, Ls, _ = next(prep(raw, sigma, 0.01, batch_size=100))
    >>> import matplotlib.pyplot as plt
    ... for i in range(len(raw)):
    ...     Y_raw, _ = raw[i]
    ...     plt.plot(Y_raw)
    ...     plt.axvline(Ls[i])

What representation to use
==========================

There are three main representations used in this library: landmarks, pre-shapes, and dual pre-shapes.
You need to choose the appropriate representation based on your analysis goals.

Landmarks
---------

Landmark coordinates, which construct the configuration matrix, are the primitive representation used for shape analysis.
They provide a direct mapping of key points on the original edge profiles.
Use this representation when you need to capture complete information from edge profiles, including shape, scale, and location.

Of course, you can preprocess the data to exclude certain information.
The following example captures pseudo-landmarks from scaled edge profiles to exclude the size variation of coating layers.

.. plot::
    :context: reset

    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge.api import scale_area
    >>> from heavyedge_landmarks import pseudo_landmarks
    >>> with ProfileData(get_sample_path("Prep-Type1.h5")) as data:
    ...     x1 = data.x()
    ...     Ys1, Ls1, _ = next(scale_area(data, batch_size=100))
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x2 = data.x()
    ...     Ys2, Ls2, _ = next(scale_area(data, batch_size=100))
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     x3 = data.x()
    ...     Ys3, Ls3, _ = next(scale_area(data, batch_size=100))
    >>> k = 10  # Number of landmarks
    >>> lm1 = pseudo_landmarks(x1, Ys1, Ls1, k)
    >>> lm2 = pseudo_landmarks(x2, Ys2, Ls2, k)
    >>> lm3 = pseudo_landmarks(x3, Ys3, Ls3, k)
    >>> import matplotlib.pyplot as plt
    ... plt.plot(x1, Ys1.T, color="gray", alpha=0.5)
    ... plt.plot(*lm1.transpose(1, 2, 0))
    ... plt.plot(x2, Ys2.T, color="gray", alpha=0.5)
    ... plt.plot(*lm2.transpose(1, 2, 0))
    ... plt.plot(x3, Ys3.T, color="gray", alpha=0.5)
    ... plt.plot(*lm3.transpose(1, 2, 0))

There are two types of landmarks: **pseudo-landmarks** and **mathematical landmarks**.

- **Pseudo-landmarks** are located by equidistant sampling. Use these when you want to analyze profiles with arbitrary shapes, or to capture the global shape of the coating layer.
- **Mathematical landmarks** are defined by specific mathematical properties. Use these when you need to analyze profiles with known geometric features.

For example, you might want to use CNNs for pseudo-landmarks but DNNs for mathematical landmarks.
All landmarks can be further transformed to pre-shapes or dual pre-shapes.

Pre-shapes
----------

Pre-shapes are a transformed representation of landmarks that removes location and size information.
They are useful for analyzing intrinsic shape properties without the influence of external factors.

Dual pre-shapes
---------------

Dual pre-shapes are a further transformation of pre-shapes that maps them back to the original space.
This representation is useful for interpreting the results of shape analysis in the context of the original data.

Locating your own landmarks
===========================

The mathematical landmarks provided by this library are just one of many ways to represent edge profiles.
You can postprocess the detected landmarks to fit your specific needs, and transform them to pre-shapes.

For instance, :func:`landmarks_type2` and :func:`landmarks_type3` do not describe the plateau region because it can be performed in various ways, e.g., using the leftmost point as a landmark or using the average height.
The following example adopts the former approach and extends the original landmarks.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import landmarks_type3
    >>> plateau = np.column_stack([np.repeat(x3[0], len(Ys3)), Ys3[:, 0]])
    >>> lm3 = landmarks_type3(x3, Ys3, Ls3, 32)
    >>> lm3_mod = np.concatenate([lm3, plateau[..., np.newaxis]], axis=-1)
    >>> plt.plot(x3, Ys3.T, color="gray", alpha=0.5)
    ... plt.plot(*lm3_mod.transpose(1, 2, 0))

The resulting configuration matrices can be transformed to pre-shapes as usual.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import preshape
    >>> ps3_mod = preshape(lm3_mod)
    >>> plt.plot(*ps3_mod.transpose(1, 2, 0))

Scaling the data
================

==========
Module API
==========

Landmark detection
==================

Acquires configuration matrices for pseudo-landmarks and mathematical landmarks.

.. autofunction:: pseudo_landmarks

.. autofunction:: landmarks_type2

.. autofunction:: landmarks_type3

Pre-shape conversion
====================

.. automodule:: heavyedge_landmarks.preshape
    :members:

Plateau fitting
===============

.. automodule:: heavyedge_landmarks.plateau
    :members:
