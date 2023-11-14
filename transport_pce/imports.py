
import sys
import matplotlib.pyplot as plt
sys.path.append('/Users/bennett/Documents/Github/MovingMesh/moving_mesh_radiative_transfer')


#from package.integrator import sampling_integrator as sint
from package.convergence_test import test_plane_pulse as tpl
#from package.convergence_test import test_source as ts
#import matplotlib.pyplot as plt
#from moving_mesh_transport.plots.plot_functions.show import show
from package.make_plots import plot_video, plot_energy, plot_quantiles
from package.poster_plots import results_getter, convergence_plotter
