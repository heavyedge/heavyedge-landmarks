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

.. note::

    To run examples in this document, install the package with ``doc`` optional dependency::

        pip install heavyedge-landmarks[doc]

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
You need to specify the standard deviation `sigma` of the Gaussian kernel for the function to smooth noise internally.

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
If you want to represent your pre-shape in the original space, use :func:`preshape_dual`.
Note that pre-shapes in the original space are rank-deficient.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import preshape_dual
    >>> dual_ps3 = preshape_dual(ps3)
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
    :context: reset

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

Which representation to use
===========================

There are three main representations used in this library: landmarks, pre-shapes, and dual pre-shapes.
You need to choose the appropriate representation based on your analysis goals.

Landmarks
---------

**Landmark** coordinates, which construct the configuration matrix, are the primitive representation used for shape analysis.
They provide a direct mapping of key points on the original edge profiles.
Use this representation when you need to capture complete information from edge profiles, including shape, scale, and location.

Of course, you can preprocess the data to exclude certain information.
The following example captures pseudo-landmarks from scaled edge profiles to exclude the size variation of coating layers.

.. plot::
    :context: close-figs

    >>> from heavyedge import ProfileData
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
    >>> pseudo_lm1 = pseudo_landmarks(x1, Ys1, Ls1, k)
    >>> pseudo_lm2 = pseudo_landmarks(x2, Ys2, Ls2, k)
    >>> pseudo_lm3 = pseudo_landmarks(x3, Ys3, Ls3, k)
    >>> import matplotlib.pyplot as plt
    ... plt.plot(x1, Ys1.T, color="gray", alpha=0.5)
    ... plt.plot(*pseudo_lm1.transpose(1, 2, 0))
    ... plt.plot(x2, Ys2.T, color="gray", alpha=0.5)
    ... plt.plot(*pseudo_lm2.transpose(1, 2, 0))
    ... plt.plot(x3, Ys3.T, color="gray", alpha=0.5)
    ... plt.plot(*pseudo_lm3.transpose(1, 2, 0))

There are two types of landmarks: **pseudo-landmarks** and **mathematical landmarks**.

- **Pseudo-landmarks** are located by equidistant sampling. Use these when you want to analyze profiles with arbitrary shapes, or to capture the global shape of the coating layer. :func:`pseudo_landmarks` locates pseudo-landmarks.
- **Mathematical landmarks** are defined by specific mathematical properties. Use these when you need to analyze profiles with known geometric features. :func:`landmarks_type2` and :func:`landmarks_type3` locate mathematical landmarks.

For example, you might want to use pseudo-landmarks for classifying arbitrary profiles using CNNs but mathematical landmarks for modeling shape variation within classes.
All landmarks can be further transformed to pre-shapes or dual pre-shapes.

Pre-shapes
----------

**Pre-shapes** are a transformed representation of landmarks that removes location and size information.
They are useful for analyzing intrinsic shape properties without the influence of external factors.
For most shape analyses, pre-shapes are what you want to work with.
:func:`preshape` transforms any configuration matrices to pre-shapes.

**Dual pre-shapes** are a further transformation of pre-shapes that maps them back to the original space.
They are introduced to assist with the visualization of pre-shapes.
Note that dual pre-shape matrices are rank-deficient, and might lead to numerical errors if you use them for analysis.
:func:`preshape_dual` maps any pre-shapes back to the original space.

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

Scaling landmarks
=================

Coating profiles have very high aspect ratios.
Since the x-coordinates have much larger scales than the y-coordinates, the shape variation along the y-axis becomes negligible if you use two-dimensional data.
As a result, you might want to scale landmarks before analysis.

Within-sample scaling
---------------------

The most straightforward approach is to scale the aspect ratio of landmarks while preserving the original shape.
You might skip this step when you are dealing with only y-coordinates as one-dimensional data, but it is essential for two-dimensional data with both x- and y-coordinates.
The following example shows min-max scaling of landmarks within each sample using :func:`minmax`.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import minmax
    >>> lm3_scaled = minmax(lm3)
    >>> plt.plot(*lm3_scaled.transpose(1, 2, 0))
    ... plt.gca().set_aspect("equal")

Between-sample scaling
----------------------

If you want to inspect the distribution of landmark positions across samples, you can apply scaling to the entire dataset.
The following example shows a pipeline which standardizes landmarks, performs PCA and then inverse-transforms the result back to the original space.

.. plot::
    :context: close-figs

    >>> from sklearn.decomposition import PCA
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import Pipeline
    >>> n = 1
    >>> pipeline_pca = Pipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("pca", PCA(n_components=n)),
    ... ])
    >>> lm3_pca = pipeline_pca.fit_transform(lm3.reshape(len(lm3), -1))
    >>> lm3_pca_inv = pipeline_pca.inverse_transform(lm3_pca).reshape(lm3.shape)
    >>> fig, axes = plt.subplots(1, 2)
    ... axes[0].plot(*lm3_scaled.transpose(1, 2, 0))
    ... axes[0].set_title("Original")
    ... axes[1].plot(*lm3_pca_inv.transpose(1, 2, 0))
    ... axes[1].set_title("Reconstructed")

Dimensionality reduction
========================

In the previous example, the dimensionality of the configuration vector was reduced by standardization and subsequent PCA.
Sometimes, you instead want to perform dimensionality reduction for pre-shapes.
Because pre-shape vectors have unit norm, they lie on a high-dimensional sphere.
To preserve this structure, you need to use Principal Nested Spheres (PNS) analysis instead of PCA.

The following example shows the result of pre-shape dimensionality reduction using PNS and PCA.
The :mod:`skpns` module is used for PNS analysis.
It can be seen that PNS preserves the original shapes better than PCA.
Also, note that pre-shapes are not standardized before dimensionality reduction because doing so will destroy the hypersphere structure.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import preshape, preshape_dual
    >>> from sklearn.decomposition import PCA
    >>> from skpns import IntrinsicPNS
    >>> lms = [
    ...     pseudo_landmarks(x1, Ys1, Ls1, k)[:, [1], :],
    ...     pseudo_landmarks(x2, Ys2, Ls2, k)[:, [1], :],
    ...     pseudo_landmarks(x3, Ys3, Ls3, k)[:, [1], :],
    ... ]
    >>> scaled_lm = np.concatenate(lms, axis=0)
    >>> ps = preshape(scaled_lm)
    >>> pns = IntrinsicPNS(n)
    >>> pns_result = pns.fit_transform(ps.reshape(len(ps), -1))
    >>> pns_inv = preshape_dual(pns.inverse_transform(pns_result).reshape(ps.shape))
    >>> pca = PCA(n)
    >>> pca_result = pca.fit_transform(ps.reshape(len(ps), -1))
    >>> pca_inv = preshape_dual(pca.inverse_transform(pca_result).reshape(ps.shape))
    >>> fig, axes = plt.subplots(1, 3)
    ... axes[0].plot(*preshape_dual(ps).transpose(1, 2, 0))
    ... axes[0].set_title("Original")
    ... axes[1].plot(*pns_inv.transpose(1, 2, 0))
    ... axes[1].set_title("PNS")
    ... axes[2].plot(*pca_inv.transpose(1, 2, 0))
    ... axes[2].set_title("PCA")
    ... for ax in axes.flat:
    ...     ax.set_axis_off()

.. note::

    When pre-shapes have small variance, the PNS result will be only marginally different from PCA because the data approximately lies on the tangent space of the hypersphere, which is linear.

==========
Module API
==========

Landmark detection
==================

Acquires configuration matrices for pseudo-landmarks and mathematical landmarks.

.. autofunction:: pseudo_landmarks

.. autofunction:: landmarks_type1

.. autofunction:: landmarks_type2

.. autofunction:: landmarks_type3

Landmark scaling
================

.. automodule:: heavyedge_landmarks.scale
    :members:

Pre-shape conversion
====================

.. automodule:: heavyedge_landmarks.preshape
    :members:

Plateau fitting
===============

.. automodule:: heavyedge_landmarks.plateau
    :members:
