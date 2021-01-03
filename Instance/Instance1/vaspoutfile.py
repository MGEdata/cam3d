import collections
import os
import re

import numpy as np
from monty.io import zopen
from pymatgen.electronic_structure.plotter import BSDOSPlotter
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.outputs import Outcar, Vasprun

####dos#####
# read vasprun.xml，get band and dos information
os.chdir(r'/share/home/skk/wcx/dielectricsPe.files/0'
)
bs_vasprun = Vasprun("/share/home/skk/wcx/dielectricsPe.files/0/vasprun.xml", parse_projected_eigen=True)
kp = bs_vasprun.structures[0]
kp = Kpoints.automatic_density(kp, 10, force_gamma=False)
kp.write_file("KPOINTS")
bs_data = bs_vasprun.get_band_structure()
#

dos_data = bs_vasprun.complete_dos
# set figure parameters, draw figure
banddos_fig = BSDOSPlotter(bs_projection='elements', dos_projection='elements',
                           )
mplt = banddos_fig.get_plot(bs=bs_data, dos=dos_data)
mplt.savefig("a.png")
#########
# outcar =Outcar(r'mp-ele-inoc/OUTCAR')
# chgcar = Chgcar.from_file(r'/home/iap13/wcx/cx_flies/0/CHGCAR')
# # xdatcar = Xdatcar(r'mp-3626-refine/XDATCAR')
#
#
# import matplotlib.pyplot as plt
# from mgetool.show import BasePlot
# import matplotlib
# matplotlib.use('Agg')
# from cams.propressing.electro import ChgCar
# elfcar = ChgCar.from_file(r'C:\Users\Administrator\Desktop/CHGCAR')
# a = elfcar.plot_contour(show_mode="show")
# b = elfcar.plot_mcontour3d(show_mode="show")
# # elfcar.plot_field(show_mode="show",vmin=0.5,vmax=1.0)
#
# data = elfcar.elf_data[:, :, 0]
# bp = BasePlot()
# bp.imshow(data)
# plt.show()
#
#
# import matplotlib.pyplot as plt
# from mgetool.show import BasePlot
#
#
# # import matplotlib
# # matplotlib.use('Agg')
# from cams.propressing.electro import ChgCar
#
# elfcar = ChgCar.from_file(r'C:\Users\Administrator\Desktop/CHGCAR')
# a = elfcar.plot_contour(show_mode="show")
# b = elfcar.plot_mcontour3d(show_mode="show")
# # elfcar.plot_field(show_mode="show",vmin=0.5,vmax=1.0)
#
# data = elfcar.elf_data[:, :, 0]
# bp = BasePlot()
# bp.imshow(data)
# plt.show()


# outcat = Outcar("/share/home/skk/wcx/test2/OUTCARp")
#
# outcat.read_freq_dielectric()
# a = outcat.dielectric_tensor_function
# a = a[:,0,0]





# piezo = outcat.read_piezo_tensor()
# lep = outcat.read_lepsilon()