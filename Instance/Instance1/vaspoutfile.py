from pymatgen.io.vasp.outputs import Chgcar

####dos#####
# # read vasprun.xml，get band and dos information
# bs_vasprun = Vasprun("vasprun.xml", parse_projected_eigen=True)
# bs_data = bs_vasprun.get_band_structure(line_mode=True)
#
# dos_vasprun = Vasprun("vasprun.xml")
# dos_data = dos_vasprun.complete_dos

# # set figure parameters, draw figure
# banddos_fig = BSDOSPlotter(bs_projection='elements', dos_projection='elements',
#                            )
# banddos_fig.get_plot(bs=bs_data, dos=dos_data)

#########
# outcar =Outcar(r'mp-ele-inoc/OUTCAR')
chgcar = Chgcar.from_file(r'/home/iap13/wcx/cx_flies/0/CHGCAR')
# xdatcar = Xdatcar(r'mp-3626-refine/XDATCAR')


import matplotlib.pyplot as plt
from mgetool.show import BasePlot


# import matplotlib
# matplotlib.use('Agg')
from cams.propressing.electro import ChgCar

elfcar = ChgCar.from_file(r'C:\Users\Administrator\Desktop/CHGCAR')
a = elfcar.plot_contour(show_mode="show")
b = elfcar.plot_mcontour3d(show_mode="show")
# elfcar.plot_field(show_mode="show",vmin=0.5,vmax=1.0)

data = elfcar.elf_data[:, :, 0]
bp = BasePlot()
bp.imshow(data)
plt.show()
