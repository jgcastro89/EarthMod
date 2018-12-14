# EarthMod
Visualization tool for geophysical data (gravity, magnetics, etc) and any 3D dataset where contourmaps, and heat maps need to be produced.
The tool can take input files with 3 columns (x,y,z) in text format or csv.
To start using the dashboard type "python EarthMod_gui.py" into a terminal window.

This tool requres the instalation of 
  pyqtgraph
  pandas
  numpy
  scipy
  pykrige
  sklearn
  matplotlib
These packages and their dependencies are best handeled by anaconda.
The following is a step by step procedure on how to install these packages.
1) install anaconda (2.7) (https://www.anaconda.com/download/)
2) install pyqtgraph through anaconda (https://anaconda.org/anaconda/pyqtgraph)
4) install sklearn through anaconda (https://anaconda.org/anaconda/scikit-learn)
5) install pykrige through anaconda (https://pykrige.readthedocs.io/en/latest/overview.html, https://anaconda.org/conda-forge/pykrige)
6) matplotlib, numpy, scipy, and pandas should already be installed with anaconda. You can check by typing "conda list" into a terminal window. 
