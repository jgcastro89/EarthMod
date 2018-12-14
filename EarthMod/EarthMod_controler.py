import pandas as pd
from pyqtgraph import QtGui

from pandas_tableview_model import PandasModel
from contour_map import contour_plot, Surface_plot, Volume_Rendering, Surface_Mesh


class Logic(QtGui.QMainWindow):
    def __init__(self):
        super(Logic, self).__init__()

        self.Contour_Map = contour_plot()
        self.Surface_Plot = Surface_plot()
        self.Surface_Mesh = Surface_Mesh()
        self.Volume_Rendering = Volume_Rendering()

        self.pandas_tableview = QtGui.QTableView()

        # variables for 2D maps
        self.x_dim = 100
        self.y_dim = 100
        self.z_dim = self.x_dim/2
        self.xyz_file_name = None
        self.pandas_dataframe = None
        self.pandas_model = None

        # buttons for menu dock
        self.set_x_grid_size = QtGui.QPushButton('Set X Grid Size')
        self.set_y_grid_size = QtGui.QPushButton('Set Y Grid Size')
        self.set_z_grid_size = QtGui.QPushButton('Set Vertical Exaggeration')
        self.open_fileButton = QtGui.QPushButton('Open File')
        self.extrapolation_mapButton = QtGui.QPushButton('Extrapolation Methods')
        self.interpolation_mapButton = QtGui.QPushButton('Interpolation Methods')
        self.select_colormapButton = QtGui.QPushButton('Colormap Options')

        # signals for methods
        self.set_x_grid_size.clicked.connect(self.get_int_attr_X)
        self.set_y_grid_size.clicked.connect(self.get_int_attr_Y)
        self.set_z_grid_size.clicked.connect(self.get_int_attr_Z)
        self.open_fileButton.clicked.connect(self.open_file)
        self.extrapolation_mapButton.clicked.connect(self.build_extrapolation_map)
        self.interpolation_mapButton.clicked.connect(self.build_interpolation_map)
        self.select_colormapButton.clicked.connect(self.select_colormap)

    def init_menu_buttons(self, dock_widget):
        """
        This method inserts menu buttons into the specified dock widget (input).
        :param dock_widget:
        :return:
        """

        dock_widget.addWidget(self.set_x_grid_size, row=0, col=0, colspan=2)
        dock_widget.addWidget(self.set_y_grid_size, row=0, col=2, colspan=2)
        dock_widget.addWidget(self.open_fileButton, row=1, col=0, colspan=4)
        dock_widget.addWidget(self.extrapolation_mapButton, row=2, col=0, colspan=2)
        dock_widget.addWidget(self.interpolation_mapButton, row=2, col=2, colspan=2)
        dock_widget.addWidget(self.select_colormapButton, row=3, col=0, colspan=4)

    def get_int_attr_X(self):
        """
        This method assigns an integer value for the x-axis grid size. The value is stored in set_x_grid_size.
        Modifications are needed for when a user clicks cancel instead of ok.
        :return:
        """
        num, ok = QtGui.QInputDialog.getInt(self, "Set Grid Size", "Enter an Integer", 300, 100)
        input = str(num)

        if num > 99 and ok:
            self.set_x_grid_size.setText(input)
            self.x_dim = num
        else:
            self.get_int_attr_X()

        self.Contour_Map.contour_ax = None

    def get_int_attr_Y(self):
        """
        This method assigns an integer value for the y-axis grid size. Tha value is stored in set_y_grid_size.
        Modifications are needed for when a user clicks cancel.
        :return:
        """
        num, ok = QtGui.QInputDialog.getInt(self, "Set Grid Size", "Enter an Integer", 300, 100)
        input = str(num)

        if num > 99 and ok:
            self.set_y_grid_size.setText(input)
            self.y_dim = num
        else:
            self.get_int_attr_Y()

        self.Contour_Map.contour_ax = None

    def get_int_attr_Z(self):
        """
        This method assigns an integer for the z-axis grid size. The values is stored in set_z_grid_size.
        This method is currently not in use. It's application might take place when developing volumetric models.
        :return:
        """
        num, ok = QtGui.QInputDialog.getInt(self, "Set Grid Size", "Enter an Integer", 100, 1)
        input = str(num)

        if num < 201 and ok:
            self.set_z_grid_size.setText(input)
            self.Surface_Plot.update_vertical_exaggeration(self.Contour_Map.Z, self.pandas_dataframe[2].min(),
                                                           int(input))
        else:
            self.get_int_attr_Z()

    def open_file(self):
        """
        A file dialog is opened to allow users to load an CSV file that contains the xyz data.
        :return:
        """
        self.Contour_Map.contour_ax = None
        self.xyz_file_name = QtGui.QFileDialog.getOpenFileName(self, 'OpenFile')
        self.build_pandas_dataframe()

    def build_pandas_dataframe(self):
        """
        Populate a pandas dataframe with a selected CSV file opened by the user.
        :return:
        """
        if 'csv' in self.xyz_file_name:
            self.pandas_dataframe = pd.read_csv(str(self.xyz_file_name), header=None)
        else:
            self.pandas_dataframe = pd.read_table(str(self.xyz_file_name), delim_whitespace=True, header=None)
        self.build_pandas_model()

    def build_extrapolation_map(self):
        """
        This method lets the user select an extrapolation scheme from a list. It then passes data, and extrapolation
        method to Contour_Map.build_2D_grid to visualize the results.
        :return:
        """

        # Contour_Map.contour_ax is set to None in order to in order to generate a new extrapolation grid.
        self.Contour_Map.contour_ax = None

        items = ("Ordinary-Kriging", "Universal-Kriging", "Rbf-linear")

        item, ok = QtGui.QInputDialog.getItem(self, "RBF/Kriging", "Extrapolation", items, 0, False)

        # If an Rbf extrapolation method is selected, we must remove the prefix.
        if 'Rbf' in str(item):
            item = str(item).split('-')
            item = item[1]
            self.Contour_Map.build_2D_grid(self.pandas_dataframe[0], self.pandas_dataframe[1], self.pandas_dataframe[2],
                                           self.x_dim, self.y_dim,
                                           interp_type='Rbf',
                                           func=item)
        else:
            item = str(item).split('-')
            item = item[0]
            self.Contour_Map.build_2D_grid(self.pandas_dataframe[0], self.pandas_dataframe[1], self.pandas_dataframe[2],
                                           self.x_dim, self.y_dim,
                                           interp_type=None,
                                           func=item)

        #self.Surface_Mesh.init_tiangular_mesh(self.Contour_Map.X, self.Contour_Map.Y, self.Contour_Map.Z,
        #                                      self.pandas_dataframe[2].min())

        self.Surface_Plot.init_surfaceMesh(self.Contour_Map.Z, self.x_dim, self.y_dim,
                                           self.pandas_dataframe[2].min())

    def build_interpolation_map(self):
        """
        This method lets the user select an interpolation cheme form a list. It the passes data, and interpolation
        method to Contour_Map.build_2D_grid to visualize the results.
        :return:
        """

        self.Contour_Map.contour_ax = None

        items = ("ordinary-kriging", "universal-kriging", "linear", "cubic")
        item, ok = QtGui.QInputDialog.getItem(self, "Interpolation", "Interpolation", items, 0, False)

        # If an Rbf extrapolation method is selected, we must remove the prefix.
        if 'Rbf' in str(item):
            item = str(item).split('-')
            item = item[1]
            self.Contour_Map.build_2D_grid(self.pandas_dataframe[0], self.pandas_dataframe[1], self.pandas_dataframe[2],
                                           self.x_dim, self.y_dim,
                                           interp_type='Rbf',
                                           func=item)
        else:
            try:
                item = str(item).split('-')
                item = item[0]
            except IndexError:
                pass
            self.Contour_Map.build_2D_grid(self.pandas_dataframe[0], self.pandas_dataframe[1], self.pandas_dataframe[2],
                                           self.x_dim, self.y_dim,
                                           interp_type=None,
                                           func=item)

        self.Surface_Plot.init_surfaceMesh(self.Contour_Map.Z, self.x_dim, self.y_dim,
                                           self.pandas_dataframe[2].min())

    def build_pandas_model(self):
        Pandas_Model_Instance = PandasModel(self.pandas_dataframe)
        self.pandas_tableview.setModel(Pandas_Model_Instance)
        self.pandas_tableview.resizeColumnsToContents()

    def select_colormap(self):
        """
        This method opens up a dialog to allow the user to select from a list of colormap options.
        If the extrapolation/interpolation method remains unchanged, Contour_Map.build_2D_grid skips over having to
        generate a new grid. It utilizes the existing grid and simply changes the colormap.
        :return:
        """

        items = (
            "YlGnBu_r", "Purples_r", "Blues_r", "Greens_r", "PuRd_r", "RdPu_r", "YlGn_r", "BuPu_r", "RdBu_r",
            "ocean", "gist_earth", "terrain", "seismic", "jet", "spectral",
            "viridis", "plasma", "inferno", "magma")
        item, ok = QtGui.QInputDialog.getItem(self, "Select a Colormap Option", "Colormaps", items, 0, False)

        self.Contour_Map.build_2D_grid(self.pandas_dataframe[0], self.pandas_dataframe[1], self.pandas_dataframe[2],
                                       self.x_dim, self.y_dim, item)

        self.Surface_Plot.update_colormap(item)
