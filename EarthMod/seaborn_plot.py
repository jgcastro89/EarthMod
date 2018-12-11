from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import sys
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")


def seabornplot():
    g = sns.FacetGrid(tips, col="sex", hue="time", palette="Set1",
                                hue_order=["Dinner", "Lunch"])
    g.map(plt.scatter, "total_bill", "tip", edgecolor="w")
    return g.fig


class MainWindow(QtGui.QMainWindow):
    send_fig = QtCore.pyqtSignal(str)

    def __init__(self):
        super(MainWindow, self).__init__()

        self.main_widget = QtGui.QWidget(self)

        self.fig = seabornplot()
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding,
                      QtGui.QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.button = QtGui.QPushButton("Button")
        self.label = QtGui.QLabel("A plot:")

        self.layout = QtGui.QGridLayout(self.main_widget)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)
        self.show()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec_())