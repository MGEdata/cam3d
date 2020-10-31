
from pymatgen.io.vasp.outputs import Chgcar,Elfcar,Vasprun,Outcar,Xdatcar
import matplotlib.pyplot as plt
import matplotlib as mpl
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.plotter import BSDOSPlotter

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
chgcar =Chgcar.from_file(r'/home/iap13/wcx/cx_flies/0/CHGCAR')
# xdatcar = Xdatcar(r'mp-3626-refine/XDATCAR')

