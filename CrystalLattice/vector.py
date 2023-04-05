from __future__ import annotations

from numpy import ndarray
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def default_2D_vec():
    return np.zeros(2)


def default_3D_vec():
    return np.zeros(3)


class VectorOperations(Enum):
    ADDITION = auto()
    MULTIPLICATION = auto()

    def __str__(self):
        if self == VectorOperations.ADDITION:
            return 'addition'
        elif self == VectorOperations.MULTIPLICATION:
            return 'dot and cross product'
        else:
            return ''


class DimensionUnmatchedError(Exception):
    '''Exeption that is raised when dimensions of vectors do not match.'''

    def __init__(self, operation: VectorOperations, dim1: int, dim2: int):
        self.msg = f'Vector operation "{operation}" is not possible between two dimensions {dim1} and {dim2}.'
        super().__init__(self.msg)


class DimensionOverflow(Exception):
    '''Exeption that is raised when dimension of vector is larger than 3.'''

    def __init__(self,  dim):
        self.msg = f'Dimension of vector: {dim} is not supported.'
        super().__init__(self.msg)


class Vector(ABC):
    @property
    @abstractmethod
    def dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def components(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def conjugate(self) -> Vector:
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: Vector) -> Vector:
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: Vector) -> Vector:
        raise NotImplementedError
    
    @abstractmethod
    def __mul__(self, other: Vector) -> tuple[float, Vector]:
        '''
        Calculates both dot and cross product of two vectors.
        0-th index is result of dot product.
        1-st index is result of cross product and is instance of Vector.
        '''
        raise NotImplementedError
    
    @abstractmethod
    def norm(self) -> float:
        '''
        Calculates Euclidean norm of a vector.

        Parameter:
        ----------
        self: instance of Vector

        Returns:
        --------
        : float
            Euclidean norm of self.
        '''
        raise NotImplementedError

    @abstractmethod
    def dot_product(self, other: Vector) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def cross_product(self, other: Vector) -> Vector:
        raise NotImplementedError



@dataclass(frozen=True)
class TwoDim_Vector(Vector):
    _dim = 2
    _components: ndarray = field(default_factory=default_2D_vec)

    @property
    def dim(self):
        return self._dim
    
    @property
    def components(self):
        return self._components

    def __post_init__(self):
        assert len(self.components) == 2, 'There should be only two components for 2D vector.'

    def conjugate(self) -> TwoDim_Vector:
        return TwoDim_Vector(np.array([num.conjugate() for num in self.components]))

    def __str__(self) -> str:
        return f'{self.dim}D Vector: {tuple(self.components)}'

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: TwoDim_Vector) -> TwoDim_Vector:
        if self.dim != other.dim:
            raise DimensionUnmatchedError(VectorOperations.ADDITION, self.dim, other.dim)
        return TwoDim_Vector(self.components + other.components)
    
    def __sub__(self, other: TwoDim_Vector) -> TwoDim_Vector:
        if self.dim != other.dim:
            raise DimensionUnmatchedError(VectorOperations.ADDITION, self.dim, other.dim)
        return TwoDim_Vector(self.components - other.components)
    
    def __div__(self, other: Vector) -> tuple[float, ThreeDim_Vector]:
        '''
        Calculates both dot and cross product of two vectors.
        0-th index is result of dot product.
        1-st index is result of cross product and is instance of Vector.

        If they have different dimensions, one with smaller dimension is promoted to larger dimension.
        Extra component is always set to 0.
        '''
        if other.dim > 3:
            raise DimensionOverflow(other.dim)
        elif isinstance(other, TwoDim_Vector):
            return (float(self.conjugate().components @ other.components),
                    ThreeDim_Vector(np.array([0, 0, np.cross(self.components, other.components)])))
        else:
            new_component = np.copy(self.components)
            new_component = np.append(new_component, np.zeros(other.dim - self.dim))
            return (float(ThreeDim_Vector(new_component).conjugate().components @ other.components),
                    ThreeDim_Vector(np.cross(self.components, other.components)))

    def get_real(self):
        return np.array([float(c) for c in self.components])
    
    def norm(self) -> float:
        '''
        Calculates Euclidean norm of a vector.

        Parameter:
        ----------
        self: instance of Vector

        Returns:
        --------
        : float
            Euclidean norm of self.
        '''
        return np.sqrt(self * self)
    
    def dot_product(self, other: Vector) -> float:
        '''
        Calculates dot of two vectors.

        If they have different dimensions, one with smaller dimension is promoted to larger dimension.
        Extra component is always set to 0.
        '''
        if other.dim > 3:
            raise DimensionOverflow(other.dim)
        elif isinstance(other, TwoDim_Vector):
            return float(self.conjugate().components @ other.components)
        else:
            new_component = np.copy(self.components)
            new_component = np.append(new_component, np.zeros(other.dim - self.dim))
            return float(ThreeDim_Vector(new_component).conjugate().components @ other.components)
    
    def cross_product(self, other: Vector) -> ThreeDim_Vector:
        '''
        Calculates cross product of two vectors.

        If they have different dimensions, one with smaller dimension is promoted to larger dimension.
        Extra component is always set to 0.
        '''
        if other.dim > 3:
            raise DimensionOverflow(other.dim)
        elif isinstance(other, TwoDim_Vector):
            return ThreeDim_Vector(np.array([0, 0, np.cross(self.components, other.components)]))
        else:
            new_component = np.copy(self.components)
            new_component = np.append(new_component, np.zeros(other.dim - self.dim))
            return ThreeDim_Vector(np.cross(self.components, other.components))

    def __mul__(self, other: Vector) -> float:
        return self.dot_product(other)
    
    def __matmul__(self, other: Vector) -> ThreeDim_Vector:
        return self.cross_product(other)



@dataclass(frozen=True)
class ThreeDim_Vector:
    _dim = 3
    _components: ndarray = field(default_factory=default_3D_vec)

    @property
    def dim(self):
        return self._dim
    
    @property
    def components(self) -> ndarray:
        return self._components
    
    def __post_init__(self):
        assert len(self.components) == 3, 'There should be only three components for 3D vector.'

    def conjugate(self) -> ThreeDim_Vector:
        return ThreeDim_Vector(np.array([num.conjugate() for num in self.components]))

    def __str__(self) -> str:
        return f'{self.dim}D Vector: {tuple(self.components)}'

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: ThreeDim_Vector) -> ThreeDim_Vector:
        if self.dim != other.dim:
            raise DimensionUnmatchedError(VectorOperations.ADDITION, self.dim, other.dim)
        return ThreeDim_Vector(self.components + other.components)
    
    def __sub__(self, other: ThreeDim_Vector) -> ThreeDim_Vector:
        if self.dim != other.dim:
            raise DimensionUnmatchedError(VectorOperations.ADDITION, self.dim, other.dim)
        return ThreeDim_Vector(self.components - other.components)
    
    def __div__(self, other: Vector) -> tuple[float, ThreeDim_Vector]:
        '''
        Calculates both dot and cross product of two vectors.
        0-th index is result of dot product.
        1-st index is result of cross product and is instance of Vector.

        If they have different dimensions, one with smaller dimension is promoted to larger dimension.
        Extra component is always set to 0.
        '''
        if other.dim > 3:
            raise DimensionOverflow(other.dim)
        elif isinstance(other, ThreeDim_Vector):
            return (float(self.conjugate().components @ other.components),
                    ThreeDim_Vector(np.cross(self.components, other.components)))
        else:
            new_component = np.copy(other.components)
            new_component = np.append(new_component, np.zeros(self.dim - other.dim))
            return (float(self.conjugate().components @ new_component),
                    ThreeDim_Vector(np.cross(self.components, other.components)))

    def get_real(self):
        return np.array([float(c) for c in self.components])

    def norm(self) -> float:
        '''
        Calculates Euclidean norm of a vector.

        Parameter:
        ----------
        self: instance of Vector

        Returns:
        --------
        : float
            Euclidean norm of self.
        '''
        return np.sqrt(self * self)

    def dot_product(self, other: Vector) -> float:
        '''
        Calculates dot product of two vectors.

        If they have different dimensions, one with smaller dimension is promoted to larger dimension.
        Extra component is always set to 0.
        '''
        if other.dim > 3:
            raise DimensionOverflow(other.dim)
        elif isinstance(other, ThreeDim_Vector):
            return float(self.conjugate().components @ other.components)
        else:
            new_component = np.copy(other.components)
            new_component = np.append(new_component, np.zeros(self.dim - other.dim))
            return float(self.conjugate().components @ new_component)
    
    def cross_product(self, other: Vector) -> ThreeDim_Vector:
        '''
        Calculates cross product of two vectors.

        If they have different dimensions, one with smaller dimension is promoted to larger dimension.
        Extra component is always set to 0.
        '''
        if other.dim > 3:
            raise DimensionOverflow(other.dim)
        elif isinstance(other, ThreeDim_Vector):
            return ThreeDim_Vector(np.cross(self.components, other.components))
        else:
            new_component = np.copy(other.components)
            new_component = np.append(new_component, np.zeros(self.dim - other.dim))
            return ThreeDim_Vector(np.cross(self.components, other.components))
    
    def __mul__(self, other):
        if isinstance(other, (Vector, TwoDim_Vector, ThreeDim_Vector)):
            return self.dot_product(other)
        else:
            if isinstance(other, (int, float, complex)):
                return ThreeDim_Vector(other * self.components)
            else:
                raise TypeError
    
    def __matmul__(self, other: Vector) -> ThreeDim_Vector:
        return self.cross_product(other)
