import numpy as np
import CrystalLattice as cl


from random import randint

# You can use pre-defined structures that is inside CrystalLattice.structures
cl.structures.BCCr.plot_both()
cl.show()

# You can make your own lattice like following.
random_lattice = cl.lattice.ThreeDim_Lattice(np.asarray([cl.vector.ThreeDim_Vector([randint(1, 3) for _ in range(3)]),
                                                         cl.vector.ThreeDim_Vector([randint(1, 3) for _ in range(3)]),
                                                         cl.vector.ThreeDim_Vector([randint(1, 3) for _ in range(3)])]))
# To plot crystal lattice, use following method on instance of ThreeDim_Lattice.
random_lattice.plot_lattice()
cl.show()  # Always put cl.show() at the end to actually see the plot.

# To plot reciprocal lattice, you can create do the following.
# Method 1. Create new variable.
reciprocal = random_lattice.get_reciprocal()
reciprocal.plot_lattice()
cl.show()

# Method 2. 
random_lattice.get_reciprocal().plot_lattice()
cl.show()

# For reciprocal lattice, you can plot both crystal lattice and reciprocal lattice by using plot_both method.
reciprocal.plot_both()
cl.show()

# You can also add button to switch between crystal lattice and reciprocal space by giving True value to parameter 'switchable'.
reciprocal.plot_both(switchable=True)
cl.show()

# You can also set 'show_single_cell' parameter to False to draw every cell defined by given points.
cl.structures.BCC.plot_lattice(5, 5, 5, show_single_cell=False)
cl.show()
