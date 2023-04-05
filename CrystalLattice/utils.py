import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d

from scipy.spatial import Voronoi


def set_2d_axes_equal(ax: plt.Axes) -> None:
    '''Make axes of 2D plot have equal scale so that circles appear as circles,
    squares as squares, etc.. 

    Parameter:
    ----------
        ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range])

    ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim([y_middle - plot_radius, y_middle + plot_radius])


def set_3d_axes_equal(ax: plt.Axes) -> None:
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Parameter:
    ----------
        ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def calc_points(cell: np.ndarray, nums: list[int], mgrid: np.ndarray=None) -> tuple[np.ndarray]:
    if len(nums) == 2:
        if mgrid is None:
            grid = np.mgrid[-nums[0] // 2 + nums[0] % 2:nums[0] // 2 + 1,
                            -nums[1] // 2 + nums[1] % 2:nums[1] // 2 + 1]
        else:
            grid = mgrid
        px, py = np.tensordot(cell, 
                              grid,
                              axes=[0, 0])
        return np.c_[px.ravel(), py.ravel()], grid
    elif len(nums) == 3:
        if mgrid is None:
            grid = np.mgrid[-nums[0] // 2 + nums[0] % 2:nums[0] // 2 + 1,
                            -nums[1] // 2 + nums[1] % 2:nums[1] // 2 + 1,
                            -nums[2] // 2 + nums[2] % 2:nums[2] // 2 + 1]
        else:
            grid = mgrid
        px, py, pz = np.tensordot(cell, 
                                  grid,
                                  axes = [0, 0])
        return np.c_[px.ravel(), py.ravel(), pz.ravel()], grid
    else:
        print('Only two and three dimensions are supported.')
        return None, None


def plot_translation_vectors(ax: plt.Axes, vectors: np.ndarray) -> plt.Axes:
    dim = vectors.shape[0]
    origins = np.asarray([[0 for _ in range(dim)] for __ in range(dim)])
    end_points = np.array(vectors)
    if dim == 2:
        ax.quiver(*origins, end_points[:, 0], end_points[:, 1], color=['r','b'], angles='xy', scale_units='xy', scale=1)
    elif dim == 3:
        ax.quiver(*origins, end_points[:, 0], end_points[:, 1], end_points[:, 2], color=['r','b','g'])
    else:
        print('Only two and three dimensions are supported.')
    return ax


def plot_2d_voronoi(voronoi: Voronoi, ax: plt.Axes, edge_color: str='orange') -> plt.Axes:
    enclosed_resgions = set()
    for region in voronoi.regions:
        region = np.asarray(region)
        if np.all(region >= 0):
            enclosed_resgions.update(set(region))
            
    for simplex in voronoi.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0) and set(simplex).issubset(enclosed_resgions):
            ax.plot(voronoi.vertices[simplex, 0], voronoi.vertices[simplex, 1], '-', color=edge_color)

    return ax

def plot_3d_voronoi(voronoi: Voronoi, ax: plt.Axes, color='orange', edge_color: str='k', alpha=.2) -> plt.Axes:
    enclosed_resgions = set()
    for region in voronoi.regions:
        region = np.asarray(region)
        if np.all(region >= 0):
            enclosed_resgions.update(set(region))

    for simplex in voronoi.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0) and set(simplex).issubset(enclosed_resgions):
            face = plt3d.art3d.Poly3DCollection([voronoi.vertices[simplex]], color=color, alpha=alpha)
            face.set_edgecolor(edge_color)
            ax.add_collection3d(face)

    return ax
