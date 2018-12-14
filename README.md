# EarthMod
Visualization tool for 3D datasets where the goal is to produce and analyze heatmaps or contour maps in 2D and 3D i.e geophysical data (gravity, magnetics, etc), topographic data, geospacial data. 
The tool can take input files with 3 columns (x,y,z) in text format or csv. Attached are 3 sample data sets that illustrate how input files should look like. 

To start using the dashboard type "python EarthMod_gui.py" into a terminal window.

This tool requres the instalation of 
  pyqtgraph,
  pandas,
  numpy,
  scipy,
  pykrige,
  sklearn, and
  matplotlib.
These packages and their dependencies are best handeled by anaconda.
The following is a step by step procedure on how to install these packages.
1) install anaconda (2.7) (https://www.anaconda.com/download/)
2) install pyqtgraph through anaconda (https://anaconda.org/anaconda/pyqtgraph)
4) install sklearn through anaconda (https://anaconda.org/anaconda/scikit-learn)
5) install pykrige through anaconda or pip (https://pykrige.readthedocs.io/en/latest/overview.html, https://anaconda.org/conda-forge/pykrige)
6) matplotlib, numpy, scipy, and pandas should already be installed with anaconda. You can check by typing "conda list" into a terminal window. 
