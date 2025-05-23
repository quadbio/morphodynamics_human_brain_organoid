"""
This work is based on Ellipsoid fit python by Aleksandr Bazhin
Source: https://github.com/aleksandrbazhin/ellipsoid_fit_python
Original License: MIT
Modifications by Gilles Gut are also licensed under MIT
Changes include: Reworked ellipsoid_fit to also provide evals and reworked the plotting function to include more options.
"""

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymeshfix
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from sklearn import metrics
from trimesh import Trimesh
from trimesh.smoothing import filter_taubin


def ellipsoid_plot(
    center,
    radii,
    rotation,
    ax,
    plot_axes=False,
    cage_color="b",
    cage_alpha=0.2,
    resolution=100,
    wireframe_stride=(2, 2),
    plot_surface=False,
    linewidth=None,
):
    """This functions plot an ellipsoid
    resolution : int, default 100
        Number of points for u,v parameter sampling
    wireframe_stride : tuple, default (2, 2)
        Stride values for wireframe plotting (rstride, cstride)
    plot_surface : bool, default False
        If True, plot as surface instead of wireframe
    linewidth : float, optional
        Line width for wireframe/axes plotting
    """
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = (
                np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
            )
    if plot_axes:
        axes = np.array(
            [[radii[0], 0.0, 0.0], [0.0, radii[1], 0.0], [0.0, 0.0, radii[2]]]
        )
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            if linewidth is not None:
                ax.plot(X3, Y3, Z3, color=cage_color, linewidth=linewidth)
            else:
                ax.plot(X3, Y3, Z3, color=cage_color)

    # plot ellipsoid
    if plot_surface:
        ax.plot_surface(x, y, z, color=cage_color, alpha=cage_alpha)
    else:
        if linewidth is not None:
            ax.plot_wireframe(
                x,
                y,
                z,
                rstride=wireframe_stride[0],
                cstride=wireframe_stride[1],
                color=cage_color,
                alpha=cage_alpha,
                linewidth=linewidth,
            )
        else:
            ax.plot_wireframe(
                x,
                y,
                z,
                rstride=wireframe_stride[0],
                cstride=wireframe_stride[1],
                color=cage_color,
                alpha=cage_alpha,
            )


def ellipsoid_fit(X):

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array(
        [
            x * x + y * y - 2 * z * z,
            x * x + z * z - 2 * y * y,
            2 * x * y,
            2 * x * z,
            2 * y * z,
            2 * x,
            2 * y,
            2 * z,
            1 - 0 * x,
        ]
    )
    d2 = np.array(x * x + y * y + z * z).T  # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array(
        [
            [v[0], v[3], v[4], v[6]],
            [v[3], v[1], v[5], v[7]],
            [v[4], v[5], v[2], v[8]],
            [v[6], v[7], v[8], v[9]],
        ]
    )

    center = np.linalg.solve(-A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1.0 / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v, evals
