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
