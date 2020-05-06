import matplotlib.tri as mtri
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import cm
from matplotlib.colors import LightSource, Normalize
from sklearn.preprocessing import MinMaxScaler
from pyqtgraph import QtGui
import pdb


class surfacePlot(QtGui.QDialog):
    def __init__(self, parent=None):
        super(surfacePlot, self).__init__(parent)

        self.surface_view = gl.GLViewWidget()
        self.colorMap = 'YlGnBu_r'
        self.xDim, self.yDim = 0, 0
        self.verticalExag = 1

        #self.surface_view.setBackgroundColor('#f8f8ff')
        self.surface_plot = None
        self.faces = None
        self.vertices = None
        self.x = None
        self.y = None
        self.z = None
        self.xGrid = None
        self.yGrid = None
        self.zMin = None
        self.colors = None

        self.zGrid = gl.GLGridItem()
        self.surface_view.addItem(self.zGrid)
        
    def init_surfaceMesh(self, Z, xDim, yDim, zMin):

        self.xDim, self.yDim, self.zMin, self.z = xDim, yDim, zMin, Z
        
        self.clear_view()
        self.createPlotGrid()
        self.set_vertical_exxageration()
        self.mesh_grid()
        self.generate_vertices()
        self.triangulate_faces()
        self.generate_colormap()

        self.build_triangular_mesh(self.xDim, self.yDim, zMin, self.vertices, self.faces, self.colors, 
                                    self.verticalExag)

    def createPlotGrid(self):
        self.zGrid = gl.GLGridItem()
        self.zGrid.scale(self.zMin / 10, self.zMin / 10, 1)
        self.surface_view.addItem(self.zGrid)

    def update_colormap(self, newColorMap):

        self.colorMap = newColorMap
        self.generate_colormap()
        self.build_triangular_mesh(self.xDim, self.yDim, self.zMin, 
                                    self.vertices, self.faces, self.colors, self.verticalExag)

    def add_gl_grid(self):

        xy_grid = gl.GLGridItem()
        xy_grid.scale(self.xDim/10, self.yDim/10, 1)
        xy_grid.setDepthValue(100)
        self.surface_view.addItem(self.xy_grid)

    def mesh_grid(self):
        """
        creates a new meshgrid using specified dimensions which 
        will be used as vertices for a triangular mesh
        :return:
        """
        x, y = np.meshgrid(np.linspace(0, self.xDim, self.xDim), np.linspace(0, self.yDim, self.yDim))

        self.x = x.flatten()
        self.y = y.flatten()

    def triangulate_faces(self):
        """
        utelizes matplotlib's Triangulation function to generate a triangular mesh from vertices
        :return:
        """
        faces = mtri.Triangulation(self.x, self.y)

        """
        x = np.take(self.x, self.nonNaNIndex[:,0])
        y = np.take(self.y, self.nonNaNIndex[:,1])
        faces = mtri.Triangulation(x, y)
        """

        self.faces = faces.triangles

    def set_vertical_exxageration(self):
        """
        sets all Nan values in z axis to a min, gl mesh items can not handle Nan values
        scales z values of surface plot by a specified integer (verticalExaggeration)
        :return:
        """
        self.z = self.z.flatten()
        self.z[np.isnan(self.z)] = self.zMin
        self.z = self.z * self.verticalExag

    def generate_vertices(self):
        """
        stacks meshgrid into vertices of x,y,z in preparation to generate a triangular mesh
        :return:
        """
        vertices = np.vstack((self.x, self.y))
        #vertices = np.take(vertices, self.nonNaNIndex).T
        self.vertices = np.vstack((vertices, self.z)).T

    def generate_colormap(self):

        try:
            """
            generate 32 contours for surface plot
            """
            cmap = cm.get_cmap(str(self.colorMap), 32)
            norm = Normalize(vmin=self.z.min(), vmax=self.z.max())
            self.colors = cmap(norm(self.vertices[:, -1]), bytes=True) / 255.0
        except ValueError:
            """
            generate a 3D heat surface using matplotlib's
            perceptual uniform sequential colormaps
            """
            cmap = cm.get_cmap(str(self.colorMap))
            norm = Normalize(vmin=self.z.min(), vmax=self.z.max())
            self.colors = cmap(norm(self.vertices[:, -1]), bytes=True) / 255.0

    def build_triangular_mesh(self, xDim, yDim, zMin, vertices, faces, colors, verticalExag):

        #self.clear_view()

        self.surface_plot = gl.GLMeshItem(vertexes=vertices, faces=faces,  smooth=False, drawFaces=True,
                                          drawEdges=False, edgeColor=(248, 248, 255, 0), vertexColors=colors)

        self.surface_plot.translate(-xDim/2, -yDim/2, -zMin*verticalExag, local=False)

        self.surface_plot.setGLOptions('opaque')

        self.surface_view.addItem(self.surface_plot)

    def clear_view(self):
        """
        remove the old surface plot
        :return:
        """
        try:
            self.surface_view.removeItem(self.surface_plot)
            self.surface_view.removeItem(self.zGrid)
        except ValueError:
            pass
