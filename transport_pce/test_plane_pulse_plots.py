from package.integrator import sampling_integrator as sint
from package.convergence_test import test_plane_pulse as tpl
import matplotlib.pyplot as plt
import csv
import numpy as np


tfinal_list = [1.0, 10.0]
c_list = [1,0.75,0.5,0.25]
for icount, itime in enumerate(tfinal_list):
    for ccount, ic in enumerate(c_list):
        with open(f'ganapol_results_t={itime}_c={ic}.csv', 'w') as file:
            res = tpl(tfinal = itime, c = ic)
            data = np.zeros((len(res[4]), 3))
            data[:,0] = res[4]
            data[:,1] = res[5]
            data[:,2] = res[6]
            # data.reshape(-1,data[0].size)
            writer = csv.writer(file)
            writer.writerow(['x','mean','standard deviation'])
            writer.writerows(data)



