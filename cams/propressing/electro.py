import copy
import logging

import numpy as np
from pymatgen.io.vasp import Elfcar, Chgcar, VolumetricData
from scipy.interpolate import interp2d

# whether pyplot installed

try:
    # import matplotlib
    # matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    plt_installed = True
except ImportError:
    print('Warning: Module matplotlib.pyplot is not installed')
    plt_installed = False

# whether mayavi installed
try:
    from mayavi import mlab

    mayavi_installed = True
except ImportError:
    mayavi_installed = False


class ElePlot:

    def __init__(self, data, angles=(90,90,90)):
        """

        Parameters
        ----------
        data: np.ndarray
        """
        self.elf_data = data
        self.grid = self.elf_data.shape

        self.__logger = logging.getLogger("vaspy.log")

    @staticmethod
    def expand_data(data, grid, widths):
        """

        Expand the data n times by widths.

        Parameters
        ----------
        data: np.ndarray
            elf or chg data.
        grid: tuple
            numpy.shape of data.
        widths: tuple of int, 3D
            number of replication on x, y, z axis
            According to Widths, the three-dimensional matrix was extended along the X, Y and Z axes.
            Examples:
                (2,1,3)
        """

        # expand grid
        widths = np.array(widths)
        expanded_grid = np.array(grid) * widths  # expanded grid
        # expand eld_data matrix
        expanded_data = copy.deepcopy(data)
        nx, ny, nz = widths
        # x axis
        added_data = copy.deepcopy(expanded_data)
        for i in range(nx - 1):
            expanded_data = np.append(expanded_data, added_data, axis=0)
        # y axis
        added_data = copy.deepcopy(expanded_data)
        for i in range(ny - 1):
            expanded_data = np.append(expanded_data, added_data, axis=1)
        # z axis
        added_data = copy.deepcopy(expanded_data)
        for i in range(nz - 1):
            expanded_data = np.append(expanded_data, added_data, axis=2)

        return expanded_data, expanded_grid

    def _contour_wrapper(self, axis_cut='z', distance=0.5, widths=(1, 1, 1)):

        # expand elf_data and grid
        elf_data, grid = self.expand_data(self.elf_data, self.grid,
                                          widths=widths)
        self.__logger.info('data shape = %s', str(elf_data.shape))
        # now cut the cube
        if abs(distance) > 1:
            raise ValueError('Distance must be between 0 and 1.')
        if axis_cut in ['X', 'x']:  # cut vertical to x axis
            nlayer = int(self.grid[0] * distance)
            z = elf_data[nlayer, :, :]
            ndim0, ndim1 = grid[2], grid[1]  # y, # z
            return ndim0, ndim1, z
        elif axis_cut in ['Y', 'y']:
            nlayer = int(self.grid[1] * distance)
            z = elf_data[:, nlayer, :]
            ndim0, ndim1 = grid[2], grid[0]  # x, z
            return ndim0, ndim1, z
        elif axis_cut in ['Z', 'z']:
            nlayer = int(self.grid[2] * distance)
            z = elf_data[:, :, nlayer]
            ndim0, ndim1 = grid[1], grid[0]  # x, y

            return ndim0, ndim1, z

    def plot_contour(self, axis_cut='z', distance=0.5,
                     show_mode='show', widths=(1, 1, 1)):

        '''
        Draw the ELF contour map.
        绘制ELF等值线图,切片.

        Parameter in kwargs
        -------------------

        axis_cut: str
            ['x', 'X', 'y', 'Y', 'z', 'Z'], axis which will be cut.
        distance: float
            (0.0 ~ 1.0), distance to origin
        show_mode: str
            'save' or 'show'
        widths: tuple of int,
            number of replication on x, y, z axis
        '''

        ndim0, ndim1, z = self._contour_wrapper(axis_cut=axis_cut, distance=distance,
                                                widths=widths)

        # do 2d interpolation
        # get slice object
        s = np.s_[0:ndim0:1, 0:ndim1:1]
        x, y = np.ogrid[s]
        self.__logger.info('z shape = %s, x shape = %s, y shape = %s',
                           str(z.shape), str(x.shape), str(y.shape))
        mx, my = np.mgrid[s]
        # use cubic 2d interpolation
        interpfunc = interp2d(x, y, z, kind='cubic')
        newx = np.linspace(0, ndim0, 600)
        newy = np.linspace(0, ndim1, 600)
        # -----------for plot3d---------------------
        ms = np.s_[0:ndim0:600j, 0:ndim1:600j]  # |
        newmx, newmy = np.mgrid[ms]  # |
        # -----------for plot3d---------------------
        newz = interpfunc(newx, newy)

        # plot 2d contour map
        fig2d_1, fig2d_2 = plt.figure(1), plt.figure(2)

        extent = [np.min(newx), np.max(newx), np.min(newy), np.max(newy)]
        ax1 = fig2d_1.add_subplot()
        img = ax1.imshow(newz, extent=extent, origin='lower')
        # coutour plot
        ax2 = fig2d_2.add_subplot()
        cs = ax2.contour(newx.reshape(-1), newy.reshape(-1), newz, 20, extent=extent)
        ax2.clabel(cs)
        plt.colorbar(mappable=img)

        # 3d plot
        fig3d = plt.figure(4, figsize=(12, 8))
        ax3d = fig3d.gca(projection='3d')
        ax3d.plot_surface(newmx, newmy, newz, cmap=plt.cm.RdBu_r)

        # save or show
        if show_mode == 'show':
            fig2d_1.show("1")
            fig2d_2.show("2")
            fig3d.show("3")
        elif show_mode == 'save':
            fig2d_1.savefig('surface2d.png', dpi=500)
            fig2d_2.savefig('contour2d.png', dpi=500)
            fig3d.savefig('surface3d.png', dpi=500)
        else:
            raise ValueError('Unrecognized show mode parameter : ' +
                             show_mode)

    def plot_mcontour3d(self, show_mode='show', **kwargs):
        '''
        use mayavi.mlab to plot 3d contour.

        Parameter
        ---------
        kwargs: {
            'maxct'   : float,max contour number,
            'nct'     : int, number of contours,
            'opacity' : float, opacity of contour,
            'widths'   : tuple of int
                        number of replication on x, y, z axis,
        }
        '''
        if not mayavi_installed:
            self.__logger.warning("Mayavi is not installed on your device.")
            return
        # set parameters
        widths = kwargs['widths'] if 'widths' in kwargs else (1, 1, 1)
        elf_data, grid = self.expand_data(self.elf_data, self.grid, widths)
        #        import pdb; pdb.set_trace()
        maxdata = np.max(elf_data)
        maxct = kwargs['maxct'] if 'maxct' in kwargs else maxdata
        # check maxct
        if maxct > maxdata:
            self.__logger.warning("maxct is larger than %f", maxdata)
        opacity = kwargs['opacity'] if 'opacity' in kwargs else 0.6
        nct = kwargs['nct'] if 'nct' in kwargs else 5
        # plot surface
        surface = mlab.contour3d(elf_data)
        # set surface attrs
        surface.actor.property.opacity = opacity
        surface.contour.maximum_contour = maxct
        surface.contour.number_of_contours = nct
        # reverse axes labels
        mlab.axes(xlabel='z', ylabel='y', zlabel='x')  # 是mlab参数顺序问题?
        mlab.outline()
        if show_mode == 'show':
            mlab.show()
        elif show_mode == 'save':
            mlab.savefig('mlab_contour3d.png')
        else:
            raise ValueError('Unrecognized show mode parameter : ' +
                             show_mode)

        return

    def plot_field(self, show_mode="show", **kwargs):
        """

        use mayavi.mlab to plot 3d field.

        Parameter
        ---------
        kwargs: {
            'vmin'   : ,min ,
            'vmax'   : ,max,
            'axis_cut': ,cut size,
            'nct'     : int, number of contours,
            'opacity' : float, opacity of contour,
            'widths'   : tuple of int
                        number of replication on x, y, z axis,
        }

        """

        if not mayavi_installed:
            self.__logger.warning("Mayavi is not installed on your device.")
            return
        # set parameters
        vmin = kwargs['vmin'] if 'vmin' in kwargs else 0.0
        vmax = kwargs['vmax'] if 'vmax' in kwargs else 1.0
        axis_cut = kwargs['axis_cut'] if 'axis_cut' in kwargs else 'z'
        nct = kwargs['nct'] if 'nct' in kwargs else 3
        widths = kwargs['widths'] if 'widths' in kwargs else (1, 1, 1)
        elf_data, grid = self.expand_data(self.elf_data, self.grid, widths)
        # create pipeline
        field = mlab.pipeline.scalar_field(elf_data)  # data source
        mlab.pipeline.volume(field, vmin=vmin, vmax=vmax)  # put data into volumn to visualize
        # cut plane
        if axis_cut in ['Z', 'z']:
            plane_orientation = 'z_axes'
        elif axis_cut in ['Y', 'y']:
            plane_orientation = 'y_axes'
        elif axis_cut in ['X', 'x']:
            plane_orientation = 'x_axes'
        cut = mlab.pipeline.scalar_cut_plane(
            field.children[0], plane_orientation=plane_orientation)
        cut.enable_contours = True  # 开启等值线显示
        cut.contour.number_of_contours = nct
        mlab.show()
        # mlab.savefig('field.png', size=(2000, 2000))
        if show_mode == 'show':
            mlab.show()
        elif show_mode == 'save':
            mlab.savefig('mlab_contour3d.png')
        else:
            raise ValueError('Unrecognized show mode parameter : ' +
                             show_mode)

        return None


class ChgCar(Chgcar, ElePlot):

    def __init__(self, poscar, data, data_aug=None):
        Chgcar.__init__(self, poscar, data, data_aug)
        self.elf_data = self.data["total"]
        ElePlot.__init__(self, data=self.elf_data)

    @classmethod
    def from_file(cls, filename):
        """
        Reads a CHGCAR file.

        :param filename: Filename
        :return: Chgcar
        """
        (poscar, data, data_aug) = VolumetricData.parse_file(filename)
        return cls(poscar, data, data_aug=data_aug)


class ElfCar(Elfcar, ElePlot):
    def __init__(self, poscar, data, data_aug=None):
        ElfCar.__init__(self, poscar, data, data_aug)
        self.elf_data = self.data["total"]
        ElePlot.__init__(self, data=self.elf_data)

    @classmethod
    def from_file(cls, filename):
        """
        Reads a CHGCAR file.

        :param filename: Filename
        :return: Chgcar
        """
        (poscar, data, data_aug) = VolumetricData.parse_file(filename)
        return cls(poscar, data, data_aug=data_aug)
