import sys

from pyqtgraph import QtGui
from pyqtgraph.dockarea import *

from EarthMod_controler import Logic


class UI(QtGui.QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        """
        Setting up UI: Widgets, background color, tabs, and windows size
        """

        """
        EMC contains all objects (maps, views, plots). Objects are first imported to EarthMod_controller 
        thenEarthMod_controler is imported here.
        """
        # instance of eEarthMod_controller
        self.EMC = Logic()

        self._2D_maps_ui = DockArea()
        self._3D_maps_ui = DockArea()

        self._2D_maps_tab_placement = None
        self._3D_maps_tab_placement = None

        # Dock area widgets that will contain our objects
        self.menu_dock = None
        self.data_df_dock = None
        self.contour_dock = None
        self.heat_dock = None
        self.variogram_dock = None

        self._3D_settings_dock = None
        self.map_img_dock = None
        self._3D_surface_dock = None

        # setting up layout
        self.vbox = QtGui.QGridLayout()
        self.tab_widget = QtGui.QTabWidget()
        self.tab_widget.setLayout(self.vbox)
        self.setCentralWidget(self.tab_widget)

        # calling methods
        self.set_up_tabs()
        self.build_2D_maps_docks()
        self.build_3D_map_docks()
        self.add_buttons_to_menu()
        self.add_objects_to_docks()

        self.setStyleSheet('QMainWindow{background-color: lightsteelblue}')
        self.setMinimumSize(1000, 700)
        self.center()
        self.show()

    def set_up_tabs(self):
        """Initiate two tabs for UI"""
        _2D_maps_tab = QtGui.QWidget()
        _3D_maps_tab = QtGui.QWidget()

        self._2D_maps_tab_placement = QtGui.QGridLayout(_2D_maps_tab)
        self._3D_maps_tab_placement = QtGui.QGridLayout(_3D_maps_tab)

        self.tab_widget.addTab(_2D_maps_tab, "2D Maps")
        self.tab_widget.addTab(_3D_maps_tab, "3D Maps")

    def build_2D_maps_docks(self):
        """
        This method handles the addition of docks to the 2D maps tab (_2D_maps_ui)
        :return:
        """
        self.menu_dock = Dock("Menu", size=(1, 2))
        self.data_df_dock = Dock("Data Dataframe", size=(1, 2))
        self.contour_dock = Dock("Contour Map", size=(2, 8))
        self.heat_dock = Dock("Raster Map", size=(2, 8))
        self.variogram_dock = Dock("Variogram Model", size=(2, 8))

        self._2D_maps_ui.addDock(self.menu_dock, 'left')
        self._2D_maps_ui.addDock(self.data_df_dock, 'right', self.menu_dock)
        self._2D_maps_ui.addDock(self.variogram_dock, 'bottom', self.menu_dock)
        self._2D_maps_ui.addDock(self.heat_dock, 'above', self.variogram_dock)
        #self._2D_maps_ui.addDock(self.heat_dock, 'above', self.variogram_dock)
        self._2D_maps_ui.addDock(self.contour_dock, 'bottom', self.data_df_dock)

        self._2D_maps_tab_placement.addWidget(self._2D_maps_ui)

    def build_3D_map_docks(self):
        """
        This method handles the addition of docks to the 3D maps tab (_3D_maps_ui)
        :return:
        """
        #self.map_img_dock = Dock("2D Map Img", size=(2, 6))
        self._3D_surface_dock = Dock("3D Map", size=(2, 6))

        self._3D_maps_ui.addDock(self._3D_surface_dock, 'left')
        #self._3D_maps_ui.addDock(self.map_img_dock, 'right', self._3D_surface_dock)

        self._3D_maps_tab_placement.addWidget(self._3D_maps_ui) 

    def add_buttons_to_menu(self):
        """
        This method inserts menu buttons into the menu_dock for _2D_maps_ui.
        :return:
        """

        self.menu_dock.addWidget(self.EMC.set_x_grid_size, row=0, col=0, colspan=2)
        self.menu_dock.addWidget(self.EMC.set_y_grid_size, row=0, col=2, colspan=2)
        self.menu_dock.addWidget(self.EMC.open_fileButton, row=1, col=0, colspan=4)
        self.menu_dock.addWidget(self.EMC.extrapolation_mapButton, row=2, col=0, colspan=2)
        self.menu_dock.addWidget(self.EMC.interpolation_mapButton, row=2, col=2, colspan=2)
        self.menu_dock.addWidget(self.EMC.select_colormapButton, row=3, col=0, colspan=4)

    def add_objects_to_docks(self):
        """
        EMC (EarthMod_controller) has created instances of our objects (maps/views/models/plots) from 
        contour_map.py and manages their logic. Here we are calling the instances of those objects that 
        have already been created and placing them in their appropriate docks.
        :return:
        """
        # 2D Maps Tab
        self.data_df_dock.addWidget(self.EMC.pandasTableView)
        self.contour_dock.addWidget(self.EMC.Contour_Map.contour_canvas)
        self.contour_dock.addWidget(self.EMC.Contour_Map.contour_toolbar)
        #self.heat_dock.addWidget(self.EMC.Contour_Map.heat_canvas)
        #self.heat_dock.addWidget(self.EMC.Contour_Map.heat_toolbar)
        self.heat_dock.addWidget(self.EMC.Contour_Map.heat_img)
        self.variogram_dock.addWidget(self.EMC.Contour_Map.variogram_canvas)
        self.variogram_dock.addWidget(self.EMC.Contour_Map.variogram_toolbar)

        # 3D Maps Tab
        #self.map_img_dock.addWidget(self.EMC.Contour_Map.heat_img)
        self._3D_surface_dock.addWidget(self.EMC.Surface_Plot.surface_view)
        self._3D_surface_dock.addWidget(self.EMC.set_z_grid_size)

    def center(self):
        """
        This method centers the tab widgets
        :return:
        """
        screen = QtGui.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)


def main():
    app = QtGui.QApplication(sys.argv)
    EarthModeling = UI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
