import sys

import matplotlib
import matplotlib.tri as mtri
import numpy as np
import numpy.ma as ma
import pyqtgraph as pg
import pyqtgraph.opengl as gl
#from sklearn.gaussian_process import GaussianProcess
from matplotlib import cm
from matplotlib.backends.backend_qt4agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import LightSource, Normalize
from matplotlib.figure import Figure
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pyqtgraph import QtGui
from scipy.interpolate import Rbf, griddata
class ContourPlot(QtGui.QDialog):
    def __init__(self, parent=None):
        super(ContourPlot, self).__init__(parent)

        self.contour_figure = Figure()
        self.contour_canvas = FigureCanvas(self.contour_figure)
        self.contour_toolbar = NavigationToolbar(self.contour_canvas, self)
        self.contour_figure.set_facecolor('ghostwhite')

        self.heat_figure = Figure()
        self.heat_canvas = FigureCanvas(self.heat_figure)
        self.heat_toolbar = NavigationToolbar(self.heat_canvas, self)
        self.heat_figure.set_facecolor('ghostwhite')

        self.variogram_figure = Figure()
        self.variogram_canvas = FigureCanvas(self.variogram_figure)
        self.variogram_toolbar = NavigationToolbar(self.variogram_canvas, self)
        self.variogram_figure.set_facecolor('ghostwhite')

        self.contour_ax = None
        self.heatMap_ax = None
        self.colormap = 'YlGnBu_r'
        self.X = None
        self.Y = None
        self.Z = None

        pg.setConfigOption('background', '#f8f8ff')
        pg.setConfigOption('foreground', 'k')
        self.colorMap = None
        self.jet_color_map()
        self.heat_img = pg.ImageView()

    def build_2D_grid(self, x, y, z, xDim, yDim, interp_type='Rbf', func='linear'):
        """
        build_2D_grid uses np.meshgrid and scipy-griddata/scipy-Rbf to generate 2D interpolated grids

        :param x: data points in the x axis (longitude)
        :param y: data points in the y axis (latitude)
        :param z: data points in the z axis (elevation/contamination, etc.)
        :param xDim: the dimension of the x axis (grid size)
        :param yDim: the dimension of the y axis (grid size)
        :param colormap: matplotlib colormap object
        :param interp_type: when this variable is active, scipy's Rbf interplation will be used
        :param func: specifies an Rbf function or griddata function (if interp_type=None)
        :return: N/A
        :calls: contourPlot, heat_map, heat_plot methods
        """
        self.cmapToRGB()
        func = str(func)

        if self.contour_ax is None:

            self.X, self.Y = np.meshgrid(np.linspace(np.min(x), np.max(x), xDim),
                                         np.linspace(np.min(y), np.max(y), yDim)
                                         )
            #self.X, self.Y = np.mgrid[np.min(x):np.max(x):xDim-1j,
            #                          np.min(y):np.max(y):yDim-1j]

            if interp_type == 'Rbf':
                # Rbf Extrapolation
                rbf_interpolation = Rbf(x, y, z, function=func, smooth=0.5, epsilon=1.)
                self.Z = rbf_interpolation(self.X, self.Y)

            elif func == 'Ordinary':
                # Ordinary Kriging Extrapolation
                self.ordinary_kriging(x, y, z, xDim, yDim)

            elif func == 'Universal':
                # Universal Kriging Extrapolation
                self.universal_kriging(x, y, z, xDim, yDim)

            elif func == 'ordinary':
                # Ordinary Kriging Interpolation
                self.ordinary_kriging(x, y, z, xDim, yDim)
                self.Convex_hull(x, y, z, xDim, yDim)

            elif func == 'universal':
                # Universal Kdriging Interpolation
                self.universal_kriging(x, y, z, xDim, yDim)
                self.Convex_hull(x, y, z, xDim, yDim)

            else:
                # Linear/Cubic Interpolation
                self.Z = griddata((x, y), z, (self.X, self.Y), method=func)
                self.Z = ma.array(self.Z, mask=np.isnan(self.Z))

        self.contour_plot(self.X, self.Y, self.Z, x, y)
        #self.heat_map(self.X, self.Y, self.Z, x, y)
        self.heat_plot(self.Z)

    def ordinary_kriging(self, x, y, z, xDim, yDim):

        items = ("linear", "power", "gaussian", "spherical", "exponential", "hole-effect")

        item, ok = QtGui.QInputDialog.getItem(self, "Kriging", "Select a Model", items, 0, False)

        if(ok):
            data = np.vstack((x, y))
            data = np.vstack((data, z)).T

            OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_function=str(item), 
                            verbose=False, enable_plotting=False)

            gridx = np.linspace(np.min(x), np.max(x), xDim)
            gridy = np.linspace(np.min(y), np.max(y), yDim)

            zVal, ss = OK.execute('grid', gridx, gridy)
            self.Z = zVal.data 

            self._display_variogram_model(OK)

    def universal_kriging(self, x, y, z, xDim, yDim):

        items = ("linear", "power", "gaussian", "spherical", "exponential")

        item, ok = QtGui.QInputDialog.getItem(self, "Kriging", "Select a Model", items, 0, False)

        if(ok):
            data = np.vstack((x, y))
            data = np.vstack((data, z)).T

            UK = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model=str(item),
                              drift_terms=['regional_linear'])

            gridx = np.linspace(np.min(x), np.max(x), xDim)
            gridy = np.linspace(np.min(y), np.max(y), yDim)

            zVals, ss = UK.execute('grid', gridx, gridy)
            self.Z = zVals.data

            self._display_variogram_model(UK)

    def Convex_hull(self, x, y, z, xDim, yDim):

        interp_Z = griddata((x, y), z, (self.X, self.Y), method='linear')

        self.Z[np.isnan(interp_Z)] = np.nan
        self.Z = ma.array(self.Z, mask=np.isnan(interp_Z))

    def contour_plot(self, X, Y, Z, x, y):
        """
        modifies contour_figure (declared at init) and creates a contour plot of the given x,y,z data

        :param X: X component of numpy meshgrid
        :param Y: Y component of numpy meshgrid
        :param Z: Z component of scipy griddata/Rbf interpolation
        :param x: data points in the x axis (longitude)
        :param y: data points in the y axis (latitude)
        :return: N/A
        """
        self.contour_figure.clear()

        # create an axis
        self.contour_ax = self.contour_figure.add_subplot(111)
        self.contour_ax.set_facecolor('lightsteelblue')

        # discards the old graph
        self.contour_ax.clear()

        # plot contour map
        self.cax1 = self.contour_ax.contourf(X, Y, Z, 32, cmap=self.colormap)
        ct = self.contour_ax.contour(X, Y, Z, 16, colors='black', linestyles='solid')
        self.contour_ax.clabel(ct, inline=1, fontsize=8)

        self.contour_figure.colorbar(self.cax1, orientation='vertical')
        self.contour_ax.scatter(x, y, c='k')
        self.contour_ax.spines['top'].set_color('ghostwhite')
        self.contour_ax.spines['right'].set_color('ghostwhite')
        self.contour_ax.spines['bottom'].set_color('ghostwhite')
        self.contour_ax.spines['left'].set_color('ghostwhite')
        self.contour_ax.tick_params(color='ghostwhite')
        self.contour_ax.grid(True, color='ghostwhite', linestyle='-', linewidth=0.3)

        # refresh canvas
        self.contour_canvas.draw()

    def heat_map(self, X, Y, Z, x, y):
        """
        modifies heat_figure (declared at init) and produces a heat map of the given x,y,z data

        :param X: X component of numpy meshgrid
        :param Y: Y component of numpy meshgrid
        :param Z: Z component of scipy griddata/Rbf interpolation
        :param x: data points in the x axis (longitude)
        :param y: data points in the y axis (latitude)
        :return: N/A
        """
        self.heat_figure.clear()

        # create an axis
        self.heatMap_ax = self.heat_figure.add_subplot(111, sharex=self.contour_ax, sharey=self.contour_ax)
        self.heatMap_ax.set_facecolor('lightsteelblue')

        # discards the old graph
        self.heatMap_ax.clear()

        # light source
        colormap = cm.get_cmap(self.colormap)
        ls = LightSource(azdeg=315, altdeg=25)
        rgb = ls.shade(np.flipud(Z), colormap, vert_exag=1, blend_mode='overlay')

        # generate contours
        ct = self.heatMap_ax.contour(X, Y, Z, 16, colors='black', linestyles='solid')
        self.heatMap_ax.clabel(ct, inline=1, fontsize=8)
        self.heatMap_ax.imshow(rgb, cmap=self.colormap, extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                                aspect='auto')
        # cax = ax.pcolormesh(X, Y, Z, cmap=self.colormap)

        # color bar
        self.heat_figure.colorbar(self.cax1, orientation='horizontal')
        self.heatMap_ax.spines['top'].set_color('ghostwhite')
        self.heatMap_ax.spines['right'].set_color('ghostwhite')
        self.heatMap_ax.spines['bottom'].set_color('ghostwhite')
        self.heatMap_ax.spines['left'].set_color('ghostwhite')
        self.heatMap_ax.tick_params(color='white')
        self.heatMap_ax.grid(True, color='ghostwhite', linestyle='-', linewidth=0.3)

        self.heat_canvas.draw()

    def _display_variogram_model(self, model):
        """Displays variogram model with the actual binned data"""
        self.variogram_figure.clear()

        # create an axis
        ax = self.variogram_figure.add_subplot(111)
        ax.set_facecolor('lightsteelblue')

        # discards the old graph
        ax.clear()

        ax.plot(model.lags, model.semivariance, 'r*')
        ax.plot(model.lags, model.variogram_function(model.variogram_model_parameters, model.lags), 'k-')
        ax.spines['top'].set_color('ghostwhite')
        ax.spines['right'].set_color('ghostwhite')
        ax.spines['bottom'].set_color('ghostwhite')
        ax.spines['left'].set_color('ghostwhite')
        ax.tick_params(color='white')
        ax.grid(True, color='ghostwhite', linestyle='-', linewidth=0.3)
        ax.set_xlabel("Lag distance")
        ax.set_ylabel("Semivariance")
        ax.set_title("Variogram Model")

        self.variogram_canvas.draw()

    def heat_plot(self, Z):

        self.heat_img.setImage(Z.T, xvals=np.linspace(0., 64., Z.shape[0]))

        self.heat_img.view.invertX(False)
        self.heat_img.view.invertY(False)
        self.heat_img.setColorMap(self.colorMap)

        north_arrow = pg.ArrowItem(angle=90, tipAngle=45, baseAngle=5, headLen=20, tailWidth=8, 
                                    tailLen=8, pen='r')
        north_arrow.setPos(15, 0)
        self.heat_img.addItem(north_arrow)

    def cmapToRGB(self):
        """
        extracts rgb values from matplotlib's colormaps to build pg.ColorMap object for pyqtgraph image item
        :return:
        """

        rgb = []

        try:
            cmap = cm.get_cmap(self.colormap, 10)
            for i in range(cmap.N):
                rgb.append(cmap(i)[:3])  # will return rgba, we take only first 3 so we get rgb

        except ValueError:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
            cmap = cm.get_cmap(self.colormap)

            for i in range(0, 10):
                k = matplotlib.colors.colorConverter.to_rgb(cmap(norm(i)))
                rgb.append(k)

        self.colorMap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 10), color=rgb)

    def jet_color_map(self):
        colorMap = [
            (0, 0, 0),
            (255, 0, 255),
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
            (255, 255, 255)
        ]

        self.colorMap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 7), color=colorMap)

class surfaceMesh(QtGui.QDialog):
    def __init__(self, parent=None):
        super(surfaceMesh, self).__init__(parent)
        """
        matplotlib impelementation of a trinagular mesh
        """

        self.mesh_figure = Figure()
        self.mesh_canvas = FigureCanvas(self.mesh_figure)
        self.mesh_toolbar = NavigationToolbar(self.mesh_canvas, self)
        self.mesh_figure.set_facecolor('ghostwhite')

        self.mesh_ax = None
        self.colormap = None
        self.X = None
        self.Y = None
        self.Z = None

    def init_tiangular_mesh(self, X, Y, Z, zMin, colormap='YlGnBu_r'):

        X = X.flatten()
        Y = Y.flatten()

        z = Z.data
        z[np.isnan(z)] = zMin
        z = z.flatten()

        # clear canvas
        self.mesh_figure.clear()

        # create an axis
        self.mesh_ax = self.mesh_figure.add_subplot(111, axisbg='lightsteelblue', projection='3d')

        # discards the old graph
        self.mesh_ax.clear()

        # generate faces from x,y
        faces = mtri.Triangulation(X, Y)

        # plot contour map
        self.mesh_ax.plot_trisurf(X, Y, faces.triangles, z, cmap=str(colormap), lw=0)

        # refresh canvas
        self.mesh_canvas.draw()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = ContourPlot()
    main.show()
    sys.exit(app.exec_())

"""
            elif func == 'Gaussian Process Regression':
        
                gpr = GaussianProcess(regr='constant', corr='cubic', theta0=0.01, thetaL=0.001, thetaU=1.0,
                                      nugget=6.1415)

                xy_stacked = np.vstack((x, y)).T
                gpr.fit(xy_stacked, z)

                krig_mesh = np.meshgrid(np.linspace(np.min(x), np.max(x), xDim),
                                        np.linspace(np.min(y), np.max(y), yDim)
                                        )
                krig_mesh = np.dstack(krig_mesh).reshape(-1, 2)
                gpr_interpolation = gpr.predict(krig_mesh)

                self.Z = griddata((np.hstack(self.X), np.hstack(self.Y)), gpr_interpolation, (self.X, self.Y),
                                  method='nearest')

                interp_Z = griddata((x, y), z, (self.X, self.Y), method='linear')
                interp_Z[np.isnan(interp_Z)] = 0

                for i in range(0, xDim):
                    for j in range(0, yDim):
                        if interp_Z[i][j] == 0:
                            self.Z[i][j] = np.nan

                self.Z = ma.array(self.Z, mask=np.isnan(self.Z))
"""
