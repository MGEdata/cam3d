import mgetool
from mgetool.show import BasePlot
import matplotlib.pyplot as plt
from cam.propressing.electro import ChgCar, ElfCar
# import matplotlib
# matplotlib.use('Agg')

elfcar =ChgCar.from_file(r'/home/iap13/wcx/cx_flies/0/ELFCAR')
a = elfcar.plot_contour(show_mode="show")
# b = elfcar.plot_mcontour3d(show_mode="save")
# elfcar.plot_mcontour(show_mode="save")

data = elfcar.elf_data[:,:,0]
bp = BasePlot()
bp.imshow(data)
plt.show()

