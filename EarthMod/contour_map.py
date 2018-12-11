import sys
import numpy as np
import numpy.ma as ma
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import QtGui
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from sklearn.gaussian_process import GaussianProcess
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import LightSource, Normalize
import matplotlib.tri as mtri


class contour_plot(QtGui.QDialog):
    def __init__(self, parent=None):
        super(contour_plot, self).__init__(parent)

        self.contour_figure = Figure()
        self.contour_canvas = FigureCanvas(self.contour_figure)
        self.contour_toolbar = NavigationToolbar(self.contour_canvas, self)
        self.contour_figure.set_facecolor('ghostwhite')

        self.heat_figure = Figure()
        self.heat_canvas = FigureCanvas(self.heat_figure)
        self.heat_toolbar = NavigationToolbar(self.heat_canvas, self)
        self.heat_figure.set_facecolor('ghostwhite')

        self.contour_ax = None
        self.colormap = None
        self.X = None
        self.Y = None
        self.Z = None

        pg.setConfigOption('background', '#f8f8ff')
        pg.setConfigOption('foreground', 'k')
        self.jet = None
        self.jet_color_map()
        self.heat_img = pg.ImageView()

    def build_2D_grid(self, x, y, z, xDim, yDim, colormap='YlGnBu_r', interp_type='Rbf', func='linear'):
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
        :calls: contour_plot, heat_map, heat_plot methods
        """

        self.colormap = str(colormap)
        self.cmapToHex()
        func = str(func)

        if self.contour_ax is None:

            self.X, self.Y = np.meshgrid(np.linspace(np.min(x), np.max(x), xDim),
                                         np.linspace(np.min(y), np.max(y), yDim)
                                         )

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
                self._convex_hull(x, y, z, xDim, yDim)

            elif func == 'universal':
                # Universal Kdriging Interpolation
                self.universal_kriging(x, y, z, xDim, yDim)
                self._convex_hull(x, y, z, xDim, yDim)

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

            else:
                # Linear/Cubic Interpolation
                self.Z = griddata((x, y), z, (self.X, self.Y), method=func)
                self.Z = ma.array(self.Z, mask=np.isnan(self.Z))

        self.contour_plot(self.X, self.Y, self.Z, x, y)
        self.heat_map(self.X, self.Y, self.Z, x, y, z)
        self.heat_plot(self.Z)

    def ordinary_kriging(self, x, y, z, xDim, yDim):

        data = np.vstack((x, y))
        data = np.vstack((data, z)).T

        OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_function='spherical', verbose=False,
                             enable_plotting=False)

        gridx = np.linspace(np.min(x), np.max(x), xDim)
        gridy = np.linspace(np.min(y), np.max(y), yDim)

        self.Z, ss = OK.execute('grid', gridx, gridy)

    def universal_kriging(self, x, y, z, xDim, yDim):

        data = np.vstack((x, y))
        data = np.vstack((data, z)).T

        UK = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                              drift_terms=['regional_linear'])

        gridx = np.linspace(np.min(x), np.max(x), xDim)
        gridy = np.linspace(np.min(y), np.max(y), yDim)

        self.Z, ss = UK.execute('grid', gridx, gridy)

    def _convex_hull(self, x, y, z, xDim, yDim):

        interp_Z = griddata((x, y), z, (self.X, self.Y), method='linear')
        interp_Z[np.isnan(interp_Z)] = 0

        for i in range(0, xDim):
            for j in range(0, yDim):
                if interp_Z[i][j] == 0:
                    self.Z[i][j] = np.nan

        self.Z = ma.array(self.Z, mask=np.isnan(self.Z))

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
        self.contour_ax = self.contour_figure.add_subplot(111, axisbg='lightsteelblue')

        # discards the old graph
        self.contour_ax.clear()

        # plot contour map
        self.contour_ax.contour(X, Y, Z, 16, colors='black', linestyles='solid')
        self.cax1 = self.contour_ax.contourf(X, Y, Z, 32, cmap=self.colormap)

        self.contour_figure.colorbar(self.cax1, orientation='horizontal')
        self.contour_ax.scatter(x, y, c='k')
        self.contour_ax.spines['top'].set_color('ghostwhite')
        self.contour_ax.spines['right'].set_color('ghostwhite')
        self.contour_ax.spines['bottom'].set_color('ghostwhite')
        self.contour_ax.spines['left'].set_color('ghostwhite')
        self.contour_ax.tick_params(color='ghostwhite')
        self.contour_ax.grid(True, color='ghostwhite', linestyle='-', linewidth=0.3)

        # refresh canvas
        self.contour_canvas.draw()

    def heat_map(self, X, Y, Z, x, y, z):
        """
        modifies heat_figure (declared at init) and produces a heat map of the given x,y,z data

        :param X: X component of numpy meshgrid
        :param Y: Y component of numpy meshgrid
        :param Z: Z component of scipy griddata/Rbf interpolation
        :param x: data points in the x axis (longitude)
        :param y: data points in the y axis (latitude)
        :param z: data points in the z axis (elevation/contamination etc.)
        :return: N/A
        """
        self.heat_figure.clear()

        # create an axis
        ax = self.heat_figure.add_subplot(111, sharex=self.contour_ax, sharey=self.contour_ax, axisbg='lightsteelblue')

        # discards the old graph
        ax.clear()

        # light source
        colormap = cm.get_cmap(self.colormap)
        ls = LightSource(azdeg=315, altdeg=25)
        rgb = ls.shade(np.flipud(Z), colormap, vert_exag=1, blend_mode='overlay')

        # generate contours
        ct = ax.contour(X, Y, Z, 16, colors='black', linestyles='solid')
        ax.clabel(ct, inline=1, fontsize=8)
        cax = ax.imshow(rgb, cmap=self.colormap, extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
        # cax = ax.pcolormesh(X, Y, Z, cmap=self.colormap)

        # color bar
        self.heat_figure.colorbar(self.cax1, orientation='horizontal')
        ax.spines['top'].set_color('ghostwhite')
        ax.spines['right'].set_color('ghostwhite')
        ax.spines['bottom'].set_color('ghostwhite')
        ax.spines['left'].set_color('ghostwhite')
        ax.tick_params(color='white')
        ax.grid(True, color='ghostwhite', linestyle='-', linewidth=0.3)

        self.heat_canvas.draw()

    def heat_plot(self, Z):

        # self.roi = pg.LineSegmentROI([[0, 50], [100, 50]], pen='r')

        self.heat_img.setImage(Z.T, xvals=np.linspace(0., 100., Z.shape[0]))

        # self.heat_img.addItem(self.roi)

        self.heat_img.view.invertX(False)
        self.heat_img.view.invertY(False)
        self.heat_img.setColorMap(self.jet)

        north_arrow = pg.ArrowItem(angle=90, tipAngle=45, baseAngle=5, headLen=20, tailWidth=8, tailLen=8, pen='r')
        north_arrow.setPos(15, 0)
        self.heat_img.addItem(north_arrow)

        # self.cross_section = pg.GraphicsWindow()
        # self.slice_plot = self.cross_section.addPlot()

        def update_cross_section():
            slice = self.roi.getArrayRegion(self.vox_model, self.geoMap.imageItem, axes=(0, 1))
            topo_slice = self.roi.getArrayRegion(self.topo, self.geoMap.imageItem, axes=(0, 1))
            bedrock_slice = self.roi.getArrayRegion(self.bedrock, self.geoMap.imageItem, axes=(0, 1))

            top = pg.PlotCurveItem(topo_slice)
            bottom = pg.PlotCurveItem(bedrock_slice)
            overburden = pg.FillBetweenItem(top, bottom, brush=(66, 134, 244, 100))

            self.slice_plot.clear()
            self.slice_plot.addItem(overburden)

            top = pg.PlotCurveItem(bedrock_slice, pen=None)
            top.setFillLevel(0.0)
            top.setBrush(57, 239, 47, 100)
            self.slice_plot.addItem(top)

            self.imv2.setImage(slice)

            # self.roi.sigRegionChanged.connect(update_cross_section)

    def cmapToHex(self):

        if self.colormap == "plasma":
            cmap = cm.plasma

        elif self.colormap == 'inferno':
            cmap = cm.inferno

        elif self.colormap == 'magma':
            cmap = cm.magma

        elif self.colormap == 'viridis':
            cmap = cm.viridis

        else:
            cmap = cm.get_cmap(self.colormap, 10)  # PiYG

        rgb = []

        try:
            for i in range(cmap.N):
                rgb.append(cmap(i)[:3])  # will return rgba, we take only first 3 so we get rgb
                # print(matplotlib.colors.rgb2hex(rgb))
        except ValueError:
            for i in range(10):
                rgb.append(cmap(i*25)[:3])

        self.jet = pg.ColorMap(pos=np.linspace(0.0, 1.0, 9), color=rgb)

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

        self.jet = pg.ColorMap(pos=np.linspace(0.0, 1.0, 7), color=colorMap)


class Surface_plot(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Surface_plot, self).__init__(parent)

        self.surface_view = gl.GLViewWidget()
        self.surface_plot = None
        self.faces = None
        self.x, self.y, self.z = None, None, None
        self.zMin = None
        self.vertices = None
        self.colors = None
        self.xDim, self.yDim = 0, 0
        self.verticalExag = 1
        self.colorMap = 'YlGnBu_r'

    def init_surfaceMesh(self, Z, xDim, yDim, zMin):

        self.xDim, self.yDim, self.zMin, self.colorMap = xDim, yDim, zMin, 'YlGnBu_r'

        self.clear_view()
        self.mesh_grid()
        self.triangulate_faces()
        self.processing_zValues(Z, zMin, self.verticalExag)
        self.generate_vertices()
        self.generate_colormap()

        self.build_triangular_mesh(self.xDim, self.yDim, zMin, self.vertices, self.faces, self.colors, self.verticalExag)

    def update_vertical_exaggeration(self, Z, zMin, verticalExaggeration):

        self.verticalExag = verticalExaggeration

        self.processing_zValues(Z, zMin, self.verticalExag)
        self.generate_vertices()
        self.generate_colormap()
        self.build_triangular_mesh(self.xDim, self.yDim, self.zMin, self.vertices, self.faces, self.colors,
                                   self.verticalExag)

    def update_colormap(self, newColorMap):

        self.colorMap = newColorMap
        self.generate_colormap()
        self.build_triangular_mesh(self.xDim, self.yDim, self.zMin, self.vertices, self.faces, self.colors, self.verticalExag)

    def add_gl_grid(self):

        xy_grid = gl.GLGridItem()
        xy_grid.scale(self.xDim/10, self.yDim/10, 1)
        xy_grid.setDepthValue(100)
        self.surface_view.addItem(self.xy_grid)

    def mesh_grid(self):

        x, y = np.meshgrid(np.linspace(0, self.xDim, self.xDim), np.linspace(0, self.yDim, self.yDim))

        self.x = x.flatten()
        self.y = y.flatten()

    def triangulate_faces(self):

        faces = mtri.Triangulation(self.x, self.y)
        self.faces = faces.triangles

    def processing_zValues(self, Z, zMin, verticalExag):

        z = Z.data
        z[np.isnan(z)] = zMin
        self.z = z.flatten()*verticalExag

    def generate_vertices(self):

        vertices = np.vstack((self.x, self.y))
        self.vertices = np.vstack((vertices, self.z)).T

    def generate_colormap(self):

        cmap = cm.get_cmap(str(self.colorMap), 31)
        norm = Normalize(vmin=self.z.min(), vmax=self.z.max())
        self.colors = cmap(norm(self.vertices[:, -1]), bytes=True)/255.0

    def build_triangular_mesh(self, xDim, yDim, zMin, vertices, faces, colors, verticalExag):

        self.clear_view()

        self.surface_plot = gl.GLMeshItem(vertexes=vertices, faces=faces,  smooth=False, drawFaces=True,
                                          drawEdges=False, edgeColor=(0, 0, 0, 0.1), vertexColors=colors,
                                          shader='shaded')

        self.surface_plot.translate(-xDim/2, -yDim/2, -zMin*verticalExag)

        self.surface_plot.setGLOptions('opaque')

        self.surface_view.addItem(self.surface_plot)

    def clear_view(self):
        try:
            self.surface_view.removeItem(self.surface_plot)
        except:
            pass


class Surface_Mesh(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Surface_Mesh, self).__init__(parent)
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


class Volume_Rendering():
    def __init__(self):

        self.xDim = None
        self.yDim = None
        self.zDim = None
        self.bedrock = None
        self.Volume = None

        self.win = gl.GLViewWidget()
        self.win.opts['distance'] = 300

        xgrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        xgrid.rotate(90, 0, 1, 0)

        self.win.addItem(xgrid)
        self.win.addItem(zgrid)

        #self.Geo_Volume()
        axis = gl.GLAxisItem()
        self.win.addItem(axis)

    def Geo_Volume(self, xDim, yDim, zDim, bedrock, z_min):

        self.xDim = xDim
        self.yDim = yDim
        self.zDim = zDim
        self.z_min = z_min
        self.bedrock = bedrock.T

        data = (self.xDim, self.yDim, self.zDim)
        data = np.zeros(data)

        self.Volume = np.empty(data.shape + (4,), dtype=np.ubyte)

        voxelModel = gl.GLVolumeItem(self.Volume, smooth=True, sliceDensity=1)
        voxelModel.translate(-(self.xDim / 2), -(self.yDim / 2), 0)
        self.win.addItem(voxelModel)

        self.clear_nan_value()

        self.update_volumetric_model()

        # return w, self.Volume

    def clear_nan_value(self):

        #self.bedrock = np.array(self.bedrock, dtype=float64)

        self.bedrock[np.isnan(self.bedrock)] = 0
        #self.bedrock[self.bedrock < 0] = 0

    def update_volumetric_model(self):
        for i in range(0, self.xDim):
            n = 0
            for j in range(0, self.yDim):
                low = int(self.bedrock[i][j])

                # Volume[i][j][0:high/7] = [57,239,47,1]
                self.Volume[i][j][0:low /self.z_min] = [57, 239, 47, 100]
                #self.Volume[0:40] = [255, 0, 0, 100]
                self.Volume[i][j][low / self.z_min:self.zDim] = [0, 0, 255, 0]

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = contour_plot()
    main.show()
    sys.exit(app.exec_())
