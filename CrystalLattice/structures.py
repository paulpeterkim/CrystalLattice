from CrystalLattice.utils import *
from CrystalLattice.vector import *
from CrystalLattice.lattice import *

SquareLattice = TwoDim_Lattice(np.asarray([TwoDim_Vector(np.asarray([1, 0])),
                                           TwoDim_Vector(np.asarray([0, 1]))]))
Hexagonal2DLattice = TwoDim_Lattice(np.asarray([TwoDim_Vector(np.asarray([1, 0])),
                                                TwoDim_Vector(np.asarray([-.5, 3 ** .5 * .5]))]))
SimpleCubic = ThreeDim_Lattice(np.asarray([ThreeDim_Vector([1, 0, 0]),
                                           ThreeDim_Vector([0, 1, 0]),
                                           ThreeDim_Vector([0, 0, 1])]))
BodyCenteredCubic = ThreeDim_Lattice(np.asarray([ThreeDim_Vector([-.5, .5, .5]),
                                                 ThreeDim_Vector([.5, -.5, .5]),
                                                 ThreeDim_Vector([.5, .5, -.5])]))
FaceCenteredCubic = ThreeDim_Lattice(np.asarray([ThreeDim_Vector([.5, .5, 0]),
                                                 ThreeDim_Vector([0, .5, .5]),
                                                 ThreeDim_Vector([.5, 0, .5])]))
Hexagonal3DLattice = ThreeDim_Lattice(np.asarray([ThreeDim_Vector([1, 0, 0]),
                                                  ThreeDim_Vector([-.5, 3 ** .5 * .5, 0]),
                                                  ThreeDim_Vector([0, 0, 1])]))

# Alias
Square = SquareLattice
D_4a = SquareLattice

# Alias
Hexagonal2D = Hexagonal2DLattice
D_6 = Hexagonal2DLattice
hp = Hexagonal2DLattice

# Alias
SC = SimpleCubic
cP = SimpleCubic

# Alias
BCC = BodyCenteredCubic
cI = BodyCenteredCubic

# Alias
FCC = FaceCenteredCubic
cF = FaceCenteredCubic

CubicLattices = [cP, cI, cF]
O_h = CubicLattices  # Alias

# Alias
Hexagonal3D = Hexagonal3DLattice
D_6h = Hexagonal3DLattice
hP = Hexagonal3DLattice

SimpleCubicReciprocal = SimpleCubic.get_reciprocal()
SCr = SimpleCubicReciprocal  #Alias

BodyCenteredCubicReciprocal = BodyCenteredCubic.get_reciprocal()
BCCr = BodyCenteredCubicReciprocal  # Alias

FaceCenteredCubicReciprocal = FaceCenteredCubic.get_reciprocal()
FCCr = FaceCenteredCubicReciprocal  # Alias

Hexagonal3DLatticeReciprocal = Hexagonal3DLattice.get_reciprocal()
Hexagonal3Dr = Hexagonal3DLatticeReciprocal  # Alias
D_6hr = Hexagonal3DLatticeReciprocal         # Alias
