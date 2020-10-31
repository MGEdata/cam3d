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


class ChgCar(Chgcar):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elf_data = self.data["total"]
        self.grid = self.elf_data.shape
        self.__logger = logging.getLogger("vaspy.log")

    @classmethod
    def from_file(cls, filename):
        """
        Reads a CHGCAR file.

        :param filename: Filename
        :return: Chgcar
        """
        (poscar, data, data_aug) = VolumetricData.parse_file(filename)
        return cls(poscar, data, data_aug=data_aug)

    @staticmethod
    def expand_data(data, grid, widths):
        '''
        根据widths, 将三维矩阵沿着x, y, z轴方向进行扩展.
        '''
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

    # 装饰器
    def contour_decorator(func):
        '''
        等值线作图方法装饰器.
        Decorator for contour plot methods.
        Set ndim on x, y axis and z values.
        '''

        def contour_wrapper(self, axis_cut='z', distance=0.5,
                            show_mode='show', widths=(1, 1, 1)):
            '''
            绘制ELF等值线图
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
                ndim0, ndim1 = grid[2], grid[1]  # y, z
            elif axis_cut in ['Y', 'y']:
                nlayer = int(self.grid[1] * distance)
                z = elf_data[:, nlayer, :]
                ndim0, ndim1 = grid[2], grid[0]  # x, z
            elif axis_cut in ['Z', 'z']:
                nlayer = int(self.grid[2] * distance)
                z = elf_data[:, :, nlayer]
                ndim0, ndim1 = grid[1], grid[0]  # x, y

            return func(self, ndim0, ndim1, z, show_mode=show_mode)

        return contour_wrapper

    @contour_decorator
    def plot_contour(self, ndim0, ndim1, z, show_mode):
        '''
        ndim0: int, point number on x-axis
        ndim1: int, point number on y-axis
        z    : 2darray, values on plane perpendicular to z axis
        '''
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
        fig3d = plt.figure(4,figsize=(12, 8))
        ax3d = fig3d.gca(projection='3d')
        ax3d.plot_surface(newmx, newmy, newz, cmap=plt.cm.RdBu_r)

        # save or show
        if show_mode == 'show':
            plt.show()
        elif show_mode == 'save':
            fig2d_1.savefig('surface2d.png', dpi=500)
            fig2d_2.savefig('contour2d.png', dpi=500)
            fig3d.savefig('surface3d.png', dpi=500)
        else:
            raise ValueError('Unrecognized show mode parameter : ' +
                             show_mode)

        return

    @contour_decorator
    def plot_mcontour(self, ndim0, ndim1, z, show_mode):
        "use mayavi.mlab to plot contour."
        if not mayavi_installed:
            self.__logger.info("Mayavi is not installed on your device.")
            return
        # do 2d interpolation
        # get slice object
        s = np.s_[0:ndim0:1, 0:ndim1:1]
        x, y = np.ogrid[s]
        mx, my = np.mgrid[s]
        # use cubic 2d interpolation
        interpfunc = interp2d(x, y, z, kind='cubic')
        newx = np.linspace(0, ndim0, 600)
        newy = np.linspace(0, ndim1, 600)
        newz = interpfunc(newx, newy)
        # mlab
        face = mlab.surf(newx, newy, newz, warp_scale=2)
        mlab.axes(xlabel='x', ylabel='y', zlabel='z')
        mlab.outline(face)
        # save or show
        if show_mode == 'show':
            mlab.show()
        elif show_mode == 'save':
            mlab.savefig('mlab_contour3d.png')
        else:
            raise ValueError('Unrecognized show mode parameter : ' +
                             show_mode)

        return

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
        "plot scalar field for elf data"
        if not mayavi_installed:
            self.__logger.warning("Mayavi is not installed on your device.")
            return
        # set parameters
        vmin = kwargs['vmin'] if 'vmin' in kwargs else 0.0
        vmax = kwargs['vmax'] if 'vmax' in kwargs else 1.0
        axis_cut = kwargs['axis_cut'] if 'axis_cut' in kwargs else 'z'
        nct = kwargs['nct'] if 'nct' in kwargs else 5
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

        return


class ElfCar(Elfcar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elf_data = self.data["total"]
        self.grid = self.elf_data.shape
        self.__logger = logging.getLogger("vaspy.log")

    @classmethod
    def from_file(cls, filename):
        """
        Reads a ELFCAR file.

        :param filename: Filename
        :return: Elfcar
        """
        (poscar, data, data_aug) = VolumetricData.parse_file(filename)
        return cls(poscar, data)

    @staticmethod
    def expand_data(data, grid, widths):
        '''
        根据widths, 将三维矩阵沿着x, y, z轴方向进行扩展.
        '''
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

    # 装饰器
    def contour_decorator(func):
        '''
        等值线作图方法装饰器.
        Decorator for contour plot methods.
        Set ndim on x, y axis and z values.
        '''

        def contour_wrapper(self, axis_cut='z', distance=0.5,
                            show_mode='show', widths=(1, 1, 1)):
            '''
            绘制ELF等值线图
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
                ndim0, ndim1 = grid[2], grid[1]  # y, z
            elif axis_cut in ['Y', 'y']:
                nlayer = int(self.grid[1] * distance)
                z = elf_data[:, nlayer, :]
                ndim0, ndim1 = grid[2], grid[0]  # x, z
            elif axis_cut in ['Z', 'z']:
                nlayer = int(self.grid[2] * distance)
                z = elf_data[:, :, nlayer]
                ndim0, ndim1 = grid[1], grid[0]  # x, y

            return func(self, ndim0, ndim1, z, show_mode=show_mode)

        return contour_wrapper

    @contour_decorator
    def plot_contour(self, ndim0, ndim1, z, show_mode):
        '''
        ndim0: int, point number on x-axis
        ndim1: int, point number on y-axis
        z    : 2darray, values on plane perpendicular to z axis
        '''
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
        fig2d_1, fig2d_2, fig2d_3 = plt.figure(), plt.figure(), plt.figure()
        ax1 = fig2d_1.add_subplot(1, 1, 1)
        extent = [np.min(newx), np.max(newx), np.min(newy), np.max(newy)]
        img = ax1.imshow(newz, extent=extent, origin='lower')
        # coutour plot
        ax2 = fig2d_2.add_subplot(1, 1, 1)
        cs = ax2.contour(newx.reshape(-1), newy.reshape(-1), newz, 20, extent=extent)
        ax2.clabel(cs)
        plt.colorbar(mappable=img)
        # contourf plot
        ax3 = fig2d_3.add_subplot(1, 1, 1)
        ax3.contourf(newx.reshape(-1), newy.reshape(-1), newz, 20, extent=extent)

        # 3d plot
        fig3d = plt.figure(figsize=(12, 8))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot_surface(newmx, newmy, newz, cmap=plt.cm.RdBu_r)

        # save or show
        if show_mode == 'show':
            plt.show()
        elif show_mode == 'save':
            fig2d_1.savefig('surface2d.png', dpi=500)
            fig2d_2.savefig('contour2d.png', dpi=500)
            fig2d_3.savefig('contourf2d.png', dpi=500)
            fig3d.savefig('surface3d.png', dpi=500)
        else:
            raise ValueError('Unrecognized show mode parameter : ' +
                             show_mode)

        return

    @contour_decorator
    def plot_mcontour(self, ndim0, ndim1, z, show_mode):
        "use mayavi.mlab to plot contour."
        if not mayavi_installed:
            self.__logger.info("Mayavi is not installed on your device.")
            return
        # do 2d interpolation
        # get slice object
        s = np.s_[0:ndim0:1, 0:ndim1:1]
        x, y = np.ogrid[s]
        mx, my = np.mgrid[s]
        # use cubic 2d interpolation
        interpfunc = interp2d(x, y, z, kind='cubic')
        newx = np.linspace(0, ndim0, 600)
        newy = np.linspace(0, ndim1, 600)
        newz = interpfunc(newx, newy)
        # mlab
        face = mlab.surf(newx, newy, newz, warp_scale=2)
        mlab.axes(xlabel='x', ylabel='y', zlabel='z')
        mlab.outline(face)
        # save or show
        if show_mode == 'show':
            mlab.show()
        elif show_mode == 'save':
            mlab.savefig('mlab_contour3d.png')
        else:
            raise ValueError('Unrecognized show mode parameter : ' +
                             show_mode)

        return

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
        "plot scalar field for elf data"
        if not mayavi_installed:
            self.__logger.warning("Mayavi is not installed on your device.")
            return
        # set parameters
        vmin = kwargs['vmin'] if 'vmin' in kwargs else 0.0
        vmax = kwargs['vmax'] if 'vmax' in kwargs else 1.0
        axis_cut = kwargs['axis_cut'] if 'axis_cut' in kwargs else 'z'
        nct = kwargs['nct'] if 'nct' in kwargs else 5
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

        return
