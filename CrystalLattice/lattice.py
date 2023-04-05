from __future__ import annotations

from CrystalLattice.utils import *
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from matplotlib.widgets import CheckButtons
from scipy.spatial import Voronoi, voronoi_plot_2d
from CrystalLattice.vector import TwoDim_Vector, ThreeDim_Vector

import scipy
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def default_2D_lattice():
    return np.array([TwoDim_Vector(np.array([1 + 0j, 0 + 0j])), TwoDim_Vector(np.array([0 + 0j, 1 + 0j]))])


def default_3D_lattice():
    return np.array([ThreeDim_Vector(np.array([1 + 0j, 0 + 0j, 0 + 0j])), 
                     ThreeDim_Vector(np.array([0 + 0j, 1 + 0j, 0 + 0j])),
                     ThreeDim_Vector(np.array([0 + 0j, 0 + 0j, 1 + 0j]))])


def default_2d_reciprocal() -> TwoDim_Lattice:
    return TwoDim_Lattice()


def default_3d_reciprocal() -> ThreeDim_Lattice:
    return ThreeDim_Lattice()


class Lattice(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def translation_vectors(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def cell(self) -> np.ndarray:
        raise NotImplementedError
    
    @cell.setter
    @abstractmethod
    def cell(self, value: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def plot_lattice(self) -> None:
        raise NotImplementedError


@dataclass
class TwoDim_Lattice(Lattice):
    _dim = 2
    _translation_vectors: np.ndarray = field(default_factory=default_2D_lattice)

    def __post_init__(self):
        assert self.translation_vectors.shape[0] == 2, 'Only 2 vectors are need for 2D lattice.'
        assert isinstance(self.translation_vectors[0], TwoDim_Vector), 'Translation vector must be two dimensional for 2D lattice.'
        assert isinstance(self.translation_vectors[1], TwoDim_Vector), 'Translation vector must be two dimensional for 2D lattice.'
        self.a1, self.a2 = self.translation_vectors
        self.area = (self.a1 @ self.a2).norm()
        self.cell = np.array([self.a1.get_real(), self.a2.get_real()])

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def translation_vectors(self) -> np.ndarray:
        return self._translation_vectors
    
    @property
    def area(self) -> float:
        return self._area

    @area.setter
    def area(self, value: float) -> None:
        self._area = value

    @property
    def a1(self) -> TwoDim_Vector:
        return self._a1

    @a1.setter
    def a1(self, value: TwoDim_Vector) -> None:
        self._a1 = value

    @property
    def a2(self) -> TwoDim_Vector:
        return self._a2

    @a2.setter
    def a2(self, value: TwoDim_Vector) -> None:
        self._a2 = value

    @property
    def cell(self) -> np.ndarray:
        return self._cell
    
    @cell.setter
    def cell(self, value: np.ndarray) -> None:
        self._cell = value

    def __str__(self) -> str:
        return f'2D Lattice of translation vectors: a1 = {self.a1.components}, a2 = {self.a2.components}'
    
    def __repr__(self) -> str:
        return self.__str__()

    def plot_lattice(self,
                     num_x: int=11, 
                     num_y: int=11, 
                     title: str='',
                     edge_color: str='orange',
                     show_single_cell: bool=False) -> tuple[plt.Figure, plt.Axes]:
        assert num_x > 0 and num_y > 0, 'num_x and num_y must be at least 1.'
        fig, ax = plt.subplots()
        a1 = self.a1.get_real()
        a2 = self.a2.get_real()

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if title:
            ax.set_tile('2D lattice: ' + title)
        else:
            ax.set_title('2D lattice: ' + r'$\mathbf{a_1} = $' + f'({a1[0]:.3f}, {a1[1]:.3f}), ' + r'$\mathbf{a_2} = $' + f'({a2[0]:.3f}, {a2[1]:.3f})')
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

        # Make all lattice point by tensor dot product.
        points, _ = calc_points(self.cell, [num_x, num_y])
        
        # Plot 2 translational vectors. 
        plot_translation_vectors(ax, self.cell)

        # Plot primitive cell.
        ax.scatter(*points.T, s=10)
        try:
            # If showing only one cell, re-define points
            if show_single_cell:
                px, py = np.tensordot(self.cell, np.mgrid[-1:2, -1:2], axes=[0, 0])
                points = np.c_[px.ravel(), py.ravel()]
            plot_2d_voronoi(Voronoi(points), ax, edge_color=edge_color)
        except scipy.spatial.qhull.QhullError:
            pass

        # Set axes as equal length.
        set_2d_axes_equal(ax)

        # Maximize window.
        plt.get_current_fig_manager().window.showMaximized()

        return fig, ax

    def get_reciprocal(self) -> TwoDim_Reciprocal:
        return TwoDim_Reciprocal(self)

    def matrix(self) -> np.ndarray:
        return np.array([self.a1.get_real(), self.a2.get_real()]).transpose()

@dataclass
class ThreeDim_Lattice(Lattice):
    _dim = 3
    _translation_vectors: np.ndarray = field(default_factory=default_3D_lattice)

    def __post_init__(self):
        assert self.translation_vectors.shape[0] == 3, 'Only 3 vectors are need for 3D lattice.'
        assert isinstance(self.translation_vectors[0], ThreeDim_Vector), 'Translation vector must be three dimensional for 3D lattice.'
        assert isinstance(self.translation_vectors[1], ThreeDim_Vector), 'Translation vector must be three dimensional for 3D lattice.'
        assert isinstance(self.translation_vectors[2], ThreeDim_Vector), 'Translation vector must be three dimensional for 3D lattice.'
        self.a1, self.a2, self.a3 = self.translation_vectors
        self.volume = self.a1 * (self.a2 @ self.a3)
        self.cell = np.array([self.a1.get_real(), self.a2.get_real(), self.a3.get_real()])

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def translation_vectors(self) -> np.ndarray:
        return self._translation_vectors
    
    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = value

    @property
    def a1(self) -> ThreeDim_Vector:
        return self._a1

    @a1.setter
    def a1(self, value: ThreeDim_Vector) -> None:
        self._a1 = value

    @property
    def a2(self) -> ThreeDim_Vector:
        return self._a2

    @a2.setter
    def a2(self, value: ThreeDim_Vector) -> None:
        self._a2 = value

    @property
    def a3(self) -> ThreeDim_Vector:
        return self._a3

    @a3.setter
    def a3(self, value: ThreeDim_Vector) -> None:
        self._a3 = value

    @property
    def cell(self) -> np.ndarray:
        return self._cell
    
    @cell.setter
    def cell(self, value: np.ndarray) -> None:
        self._cell = value

    def __str__(self) -> str:
        return f'{self.dim}D Lattice of translation vectors: a1 = {self.a1.components}, a2 = {self.a2.components}, a3 = {self.a3.components}'
    
    def __repr__(self) -> str:
        return self.__str__()

    def plot_lattice(self, 
                     num_x: int=3, 
                     num_y: int=3, 
                     num_z: int=3,
                     title: str='',
                     color: str='orange',
                     edge_color: str='k',
                     alpha: float=.2,
                     show_single_cell: bool=True) -> tuple[plt.Figure, plt.Axes]:
        assert num_x > 0 and num_y > 0 and num_z > 0, 'num_x, num_y, num_z must all be at least 1.'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        a1 = self.a1.get_real()
        a2 = self.a2.get_real()
        a3 = self.a3.get_real()

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if title:
            ax.set_title('3D lattice: ' + title, wrap=True)
        else:
            ax.set_title('3D lattice\n'\
                         + r'$\mathbf{a_1} = $' + f'({a1[0]:.3f}, {a1[1]:.3f}, {a1[2]:.3f}), '\
                         + r'$\mathbf{a_2} = $' + f'({a2[0]:.3f}, {a2[1]:.3f}, {a2[2]:.3f}), '\
                         + r'$\mathbf{a_3} = $' + f'({a3[0]:.3f}, {a3[1]:.3f}, {a3[2]:.3f})',
                         wrap=True)

        # Make all lattice point by tensor dot product.
        points, _ = calc_points(self.cell, [num_x, num_y, num_z])

        # Plot 3 translation vectors.
        plot_translation_vectors(ax, self.cell)

        # Plot primitive cell (Wigner-Seitz Cell) of given lattice.
        ax.scatter(*points.T)
        try:
            # If showing only one cell, re-define points variable.
            if show_single_cell:
                px, py, pz = np.tensordot(self.cell, np.mgrid[-1:2, -1:2, -1:2], axes = [0, 0])
                points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

            plot_3d_voronoi(Voronoi(points),
                            ax,
                            edge_color=edge_color,
                            alpha=alpha)
        except scipy.spatial.qhull.QhullError:
            pass

        # Set axes as equal length.
        set_3d_axes_equal(ax)
        
        # Maximize window.
        plt.get_current_fig_manager().window.showMaximized()

        return fig, ax
    
    def get_reciprocal(self) -> ThreeDim_Reciprocal:
        return ThreeDim_Reciprocal(self)


@dataclass
class TwoDim_Reciprocal(Lattice):
    _dim = 2
    lattice: TwoDim_Lattice = field(default_factory=default_2d_reciprocal)

    def __post_init__(self):
        self.translation_vectors = np.array([TwoDim_Vector(vec) for vec in (2 * np.pi * la.inv(self.lattice.matrix())).transpose()])
        self.b1, self.b2 = self.translation_vectors
        self.area = (self.b1 @ self.b2).norm()
        self.cell = np.array([self.b1.get_real(), self.b2.get_real()])
    
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def translation_vectors(self) -> np.ndarray:
        return self._translation_vectors

    @translation_vectors.setter
    def translation_vectors(self, value: np.ndarray) -> None:
        self._translation_vectors = value

    @property
    def area(self) -> float:
        return self._area
    
    @area.setter
    def area(self, value: float) -> None:
        self._area = value

    @property
    def b1(self) -> TwoDim_Vector:
        return self._b1

    @b1.setter
    def b1(self, value: TwoDim_Vector) -> None:
        self._b1 = value

    @property
    def b2(self) -> TwoDim_Vector:
        return self._b2

    @b2.setter
    def b2(self, value: TwoDim_Vector) -> None:
        self._b2 = value

    @property
    def cell(self) -> np.ndarray:
        return self._cell
    
    @cell.setter
    def cell(self, value: np.ndarray) -> None:
        self._cell = value

    def __str__(self) -> str:
        return f'2D Reciprocal lattice of translation vectors: b1 = {self.b1.components}, b2 = {self.b2.components}'

    def __repr__(self) -> str:
        return self.__str__()

    def plot_lattice(self, 
                     num_x: int=10,
                     num_y: int=10,
                     title: str='',
                     edge_color: str='orange') -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        b1 = self.b1.get_real()
        b2 = self.b2.get_real()

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if title:
            ax.set_tile('2D Reciprocal lattice: ' + title)
        else:
            ax.set_title('2D Reciprocal lattice: ' + r'$\mathbf{a_1} = $' + f'({b1[0]:.3f}, {b1[1]:.3f}), ' + r'$\mathbf{a_2} = $' + f'({b2[0]:.3f}, {b2[1]:.3f})')

        # Make all lattice point by tensor dot product.
        points, _ = calc_points(self.cell, [num_x, num_y])

        # Plot 1st Brillouin Zone of lattice point at origin.
        ax.scatter(*points.T, s=10)
        try:
            bzx, bzy = np.tensordot(self.cell, np.mgrid[-1:2, -1:2], axes=[0, 0])
            bz_points = np.c_[bzx.ravel(), bzy.ravel()]
            plot_2d_voronoi(Voronoi(bz_points), ax=ax, edge_color=edge_color)
        except scipy.spatial.qhull.QhullError:
            pass
            
        # Plot 2 reciprocal lattice vectors.
        plot_translation_vectors(ax, self.cell)

        # Set axes as equal length.
        set_2d_axes_equal(ax)

        # Maximize window.
        plt.get_current_fig_manager().window.showMaximized()

        return fig, ax

    def plot_both(self, 
                  num_x: int=10, 
                  num_y: int=10, 
                  title: str='',
                  edge_color: str='orange',
                  switchable: bool=False) -> tuple[plt.Figure, plt.Axes]:
        b1 = self.b1.get_real()
        b2 = self.b2.get_real()

        # Make all crystal lattice points by tensor dot products.        
        lattice_points, mgrid = calc_points(self.lattice.cell, [num_x, num_y])

        # Make all reciprocal lattice points by tensor dot products.
        reciprocal_points, _ = calc_points(self.cell, [num_x, num_y], mgrid=mgrid)

        # Make points for calculating the first brillouin zone.
        brillouin_points, _ = calc_points(self.cell, [num_x, num_y], mgrid=np.mgrid[-1:2, -1:2])

        # Calculate Wigner-Seitz cell and first Brillouin Zone using Voronoi diagram.
        try:
            Wigner_Seitz = Voronoi(lattice_points)
        except scipy.spatial.qhull.QhullError:
            Wigner_Seitz = None
        try:
            Brillouin = Voronoi(brillouin_points)
        except scipy.spatial.qhull.QhullError:
            Brillouin = None

        # Calculate origins and end points of 2 lattice translation vectors and reciprocal translation vectors. 
        origins = np.array([[0, 0], [0, 0]])
        end_points = np.array(self.lattice.cell)
        end_points_r = np.array(self.cell)

        # If switchable, plot crystal lattice, then when checked, plot reciprocal lattice.
        if switchable:
            fig, ax = plt.subplots()

            # Adjust plot position for button.
            plt.subplots_adjust(bottom=0.25)

            # Add axis for the button.
            scale_ax = fig.add_axes([0.35, 0.05, 0.3, 0.075])

            # Create button.
            switch_btn = CheckButtons(scale_ax, ['Reciprocal Lattice'], [False])

            ax.set_xlabel('x')
            ax.set_ylabel('y')

            # Plot Crystal lattice first.
            ax.set_title('Crystal Lattice')
            ax.scatter(*lattice_points.T, s=10)

            if Wigner_Seitz:
                plot_2d_voronoi(Wigner_Seitz, ax, edge_color=edge_color)

            plot_translation_vectors(ax, self.lattice.cell)

            # Callback function for the button. This will switch between view modes.
            def checked(event):
                reciprocal = switch_btn.get_status()[0]
                ax.clear()
                if reciprocal:
                    ax.set_title('Reciprocal Lattice')
                    ax.scatter(*reciprocal_points.T, s=10)
                    
                    if Brillouin:
                        plot_2d_voronoi(Brillouin, ax, edge_color=edge_color)

                    ax.quiver(*origins, end_points_r[:, 0], end_points_r[:, 1], color=['r','b'], angles='xy', scale_units='xy', scale=1)
                else:
                    ax.set_title('Crystal Lattice')
                    ax.scatter(*lattice_points.T, s=10)

                    if Wigner_Seitz:
                        plot_2d_voronoi(Wigner_Seitz, ax, edge_color=edge_color)

                    ax.quiver(*origins, end_points[:, 0], end_points[:, 1], color=['r','b'], angles='xy', scale_units='xy', scale=1)
                fig.canvas.draw_idle()

            switch_btn.on_clicked(checked)

            # Maximize window.
            plt.get_current_fig_manager().window.showMaximized()

            plt.show()

            return fig, ax
        else:
            fig, axes = plt.subplots(2)
            
            for ax in axes:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            if title:
                fig.suptitle('2D: ' + title)
            else:
                fig.suptitle(r'$\mathbf{b_1} = $' + f'({b1[0]:.3f}, {b1[1]:.3f}), ' + r'$\mathbf{b_2} = $' + f'({b2[0]:.3f}, {b2[1]:.3f})')

            axes[0].set_title('Lattice')
            axes[1].set_title('Reciprocal Lattice')

            axes[0].scatter(*lattice_points.T, s=10)
            if Wigner_Seitz:
                plot_2d_voronoi(Wigner_Seitz, ax=axes[0], edge_color=edge_color)

            axes[1].scatter(*reciprocal_points.T, s=10)
            if Brillouin:
                plot_2d_voronoi(Brillouin, ax=axes[1], edge_color=edge_color)
    
            axes[0].quiver(*origins, end_points[:, 0], end_points[:, 1], color=['r','b'], angles='xy', scale_units='xy', scale=1)
            axes[1].quiver(*origins, end_points_r[:, 0], end_points_r[:, 1], color=['r','b'], angles='xy', scale_units='xy', scale=1)

            # Set axes as equal length.
            for ax in axes:
                set_2d_axes_equal(ax)

            # Maximize window.
            plt.get_current_fig_manager().window.showMaximized()

            return fig, axes
    
    def get_lattice(self) -> TwoDim_Lattice:
        return self.lattice


@dataclass
class ThreeDim_Reciprocal(Lattice):
    _dim = 2
    lattice: TwoDim_Lattice = field(default_factory=default_3d_reciprocal)

    def __post_init__(self):
        self.translation_vectors = np.asarray([(ai @ aj) * 2 * (np.pi / self.lattice.volume)
                                               for ai, aj in zip(self.lattice.translation_vectors, 
                                                                 self.lattice.translation_vectors[[1, 2, 0]])])
        self.b1, self.b2, self.b3 = self.translation_vectors
        self.volume = self.b1 * (self.b2 @ self.b3)
        self.cell = np.array([self.b1.get_real(), self.b2.get_real(), self.b3.get_real()])

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def translation_vectors(self) -> np.ndarray:
        return self._translation_vectors
    
    @translation_vectors.setter
    def translation_vectors(self, value: np.ndarray) -> None:
        self._translation_vectors = value

    @property
    def volume(self) -> float:
        return self._volume
    
    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = value

    @property
    def b1(self) -> ThreeDim_Vector:
        return self._b1
    
    @b1.setter
    def b1(self, value: ThreeDim_Vector) -> None:
        self._b1 = value

    @property
    def b2(self) -> ThreeDim_Vector:
        return self._b2

    @b2.setter
    def b2(self, value: ThreeDim_Vector) -> None:
        self._b2 = value

    @property
    def b3(self) -> ThreeDim_Vector:
        return self._b3
    
    @b3.setter
    def b3(self, value:ThreeDim_Vector) -> None:
        self._b3 = value

    @property
    def cell(self) -> np.ndarray:
        return self._cell
    
    @cell.setter
    def cell(self, value: np.ndarray) -> None:
        self._cell = value

    def __str__(self) -> str:
        return f'3D Reciprocal lattice of translation vectors: b1 = {self.b1.components}, b2 = {self.b2.components}, b3 = {self.b3.components}'

    def __repr__(self) -> str:
        return self.__str__()
    
    def plot_lattice(self,
                     num_x: int=5,
                     num_y: int=5,
                     num_z: int=5,
                     title: str='',
                     color: str='orange',
                     edge_color: str='k',
                     alpha: float=.2) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        b1, b2, b3 = self.cell

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if title:
            ax.set_tile('3D Reciprocal lattice: ' + title)
        else:
            ax.set_title('3D Reciprocal lattice\n'\
                         + r'$\mathbf{b_1} = $' + f'({b1[0]:.3f}, {b1[1]:.3f}, {b1[2]:.3f}), '\
                         + r'$\mathbf{b_2} = $' + f'({b2[0]:.3f}, {b2[1]:.3f}, {b2[2]:.3f}), '\
                         + r'$\mathbf{b_3} = $' + f'({b3[0]:.3f}, {b3[1]:.3f}, {b3[2]:.3f})',
                         wrap=True)
            
        # Make all lattice point by tensor dot product.
        points, _ = calc_points(self.cell, [num_x, num_y, num_z])

        # Plot primitive cell (Wigner-Seitz Cell) of given lattice.
        ax.scatter(*points.T, s=10)
        try:
            bzx, bzy, bzz = np.tensordot(self.cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
            bz_points = np.c_[bzx.ravel(), bzy.ravel(), bzz.ravel()]

            plot_3d_voronoi(Voronoi(bz_points), ax, color=color, edge_color=edge_color, alpha=alpha)
        except scipy.spatial.qhull.QhullError:
            pass

        # Plot 3 translation vectors.
        plot_translation_vectors(ax, self.cell)

        # Set axes as equal length.
        set_3d_axes_equal(ax)
        
        # Maximize window.
        plt.get_current_fig_manager().window.showMaximized()

        return fig, ax
    
    def plot_both(self,
                  num_x: int=3,
                  num_y: int=3,
                  num_z: int=3,
                  title: str='',
                  color: str='orange',
                  edge_color: str='k',
                  alpha=.2,
                  show_single_cell: bool=True,
                  switchable: bool=False) -> tuple[plt.Figure, plt.Axes]:
        b1, b2, b3 = self.cell

        lattice_points, mgrid = calc_points(self.lattice.cell, [num_x, num_y, num_z])

        # Make all reciprocal lattice points by tensor dot products.
        reciprocal_points, _ = calc_points(self.cell, [num_x, num_y, num_z], mgrid=mgrid)

        # Make points for calculating the first brillouin zone.
        brillouin_points, grid = calc_points(self.cell, [num_x, num_y, num_z], mgrid=np.mgrid[-1:2, -1:2, -1:2])

        # Calculate Wigner-Seitz cell and first Brillouin Zone using Voronoi diagram.
        try:
            if show_single_cell:
                points = calc_points(self.lattice.cell, [num_x, num_y, num_z], mgrid=grid)[0]
            else:
                points = lattice_points
            Wigner_Seitz = Voronoi(points)
        except scipy.spatial.qhull.QhullError:
            Wigner_Seitz = None
        try:
            Brillouin = Voronoi(brillouin_points)
        except scipy.spatial.qhull.QhullError:
            Brillouin = None

        # Calculate origins and end points of 2 lattice translation vectors and reciprocal translation vectors. 
        origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        end_points = np.array(self.lattice.cell)
        end_points_r = np.array(self.cell)

        # If switchable, plot crystal lattice, then when checked, plot reciprocal lattice.
        if switchable:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Adjust plot position for button.
            plt.subplots_adjust(bottom=0.25)

            # Add axis for the button.
            scale_ax = fig.add_axes([0.35, 0.05, 0.3, 0.075])

            # Create button.
            switch_btn = CheckButtons(scale_ax, ['Reciprocal Lattice'], [False])

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            # Plot Crystal lattice first.
            ax.set_title('Crystal Lattice')
            ax.scatter(*lattice_points.T, s=10)

            if Wigner_Seitz:
                plot_3d_voronoi(Wigner_Seitz, ax, color=color, edge_color=edge_color, alpha=alpha)

            ax.quiver(*origins, end_points[:, 0], end_points[:, 1], end_points[:, 2], color=['r','b', 'g'])

            set_3d_axes_equal(ax)

            # Callback function for the button. This will switch between view modes.
            def checked(event):
                reciprocal = switch_btn.get_status()[0]
                ax.clear()
                if reciprocal:
                    ax.set_title('Reciprocal Lattice')
                    ax.scatter(*reciprocal_points.T, s=10)
                    
                    if Brillouin:
                        plot_3d_voronoi(Brillouin, ax, color=color, edge_color=edge_color, alpha=alpha)

                    ax.quiver(*origins, end_points_r[:, 0], end_points_r[:, 1], end_points_r[:, 2], color=['r','b', 'g'])
                else:
                    ax.set_title('Crystal Lattice')
                    ax.scatter(*lattice_points.T, s=10)

                    if Wigner_Seitz:
                        plot_3d_voronoi(Wigner_Seitz, ax, color=color, edge_color=edge_color, alpha=alpha)
                            
                    ax.quiver(*origins, end_points[:, 0], end_points[:, 1], end_points[:, 2], color=['r','b', 'g'])
                set_3d_axes_equal(ax)
                fig.canvas.draw_idle()

            switch_btn.on_clicked(checked)

            # Maximize window.
            plt.get_current_fig_manager().window.showMaximized()

            plt.show()

            return fig, ax
        else:
            fig = plt.figure()
            axes = list()
            axes.append(fig.add_subplot(121, projection='3d'))
            axes.append(fig.add_subplot(122, projection='3d'))
            
            for ax in axes:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
            if title:
                fig.suptitle('3D: ' + title)
            else:
                fig.suptitle(r'$\mathbf{b_1} = $' + f'({b1[0]:.3f}, {b1[1]:.3f}, {b1[2]:.3f}), '\
                             + r'$\mathbf{b_2} = $' + f'({b2[0]:.3f}, {b2[1]:.3f}, {b2[2]:.3f})'\
                             + r'$\mathbf{b_3} = $' + f'({b3[0]:.3f}, {b3[1]:.3f}, {b3[2]:.3f})')

            axes[0].set_title('Lattice')
            axes[1].set_title('Reciprocal Lattice')

            # Plot crystal lattice
            axes[0].scatter(*lattice_points.T, s=10)
            if Wigner_Seitz:
                plot_3d_voronoi(Wigner_Seitz, axes[0], color=color, edge_color=edge_color, alpha=alpha)

            axes[0].quiver(*origins, end_points[:, 0], end_points[:, 1], end_points[:, 2], color=['r','b', 'g'])

            # Plot reciprocal lattice
            axes[1].scatter(*reciprocal_points.T, s=10)
            if Brillouin:
                plot_3d_voronoi(Brillouin, axes[1], color=color, edge_color=edge_color, alpha=alpha)

            axes[1].quiver(*origins, end_points_r[:, 0], end_points_r[:, 1], end_points_r[:, 2], color=['r','b', 'g'])

            # Set axes as equal length.
            for ax in axes:
                set_3d_axes_equal(ax)

            # Maximize window.
            plt.get_current_fig_manager().window.showMaximized()

            return fig, axes
        
    def get_lattice(self) -> ThreeDim_Lattice:
        return self.lattice


def test_2d_lattice():
    # 2D hexagonal lattice
    a1 = TwoDim_Vector(np.array([.5, 3 ** .5 * .5]))
    a2 = TwoDim_Vector(np.array([-.5, 3 ** .5 * .5]))

    twodim = TwoDim_Lattice(np.array([a1, a2]))
    fig, ax = twodim.plot_lattice()
    fig, ax = twodim.plot_lattice(show_single_cell=True)
    plt.show()


def test_3d_lattice():
    # Primitive cell of bcc
    a1 = ThreeDim_Vector(np.array([-.5, .5, .5]))
    a2 = ThreeDim_Vector(np.array([.5, -.5, .5]))
    a3 = ThreeDim_Vector(np.array([.5, .5, -.5]))

    threedim = ThreeDim_Lattice(np.asarray([a1, a2, a3]))
    fig, ax = threedim.plot_lattice(show_single_cell=False)
    fig, ax = threedim.plot_lattice(3, 3, 3)
    plt.show()


def test_2d_reciprocal():
    a1 = TwoDim_Vector(np.array([3 ** .5, 1]))
    a2 = TwoDim_Vector(np.array([3 ** .5, -1]))
    twodim = TwoDim_Lattice(np.array([a1, a2]))
    test = TwoDim_Reciprocal(twodim)
    test.plot_lattice(10, 10)
    test.plot_both(10, 10)
    test.plot_both(11, 11, switchable=True)
    plt.show()


def test_3d_reciprocal():
    a1 = ThreeDim_Vector(np.asarray([3 ** .5, 1, 0]))
    a2 = ThreeDim_Vector(np.asarray([3 ** .5, -1, 0]))
    a3 = ThreeDim_Vector(np.asarray([0, 0, 1]))
    threedim = ThreeDim_Lattice(np.array([a1, a2, a3]))
    test = ThreeDim_Reciprocal(threedim)
    test.plot_lattice()
    test.plot_both()
    test.plot_both(switchable=True)
    plt.show()
