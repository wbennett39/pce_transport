import sys
import matplotlib.pyplot as plt
sys.path.append('/Users/bennett/Documents/Github/MovingMesh/moving_mesh_radiative_transfer')

from moving_mesh_transport.benchmarks.benchmarks import make_benchmark
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def sampling_integrator(tfinal, x0, t0, c, a1_old, a2, a3, source_name, npnts, nruns):
    plt.ion()
    result_list = np.zeros((nruns + 1, npnts))
    integrating_class = make_benchmark(source_name, x0, t0, x0, c)
    integrating_class.PCE_moments_ganapol(tfinal, npnts, a1_old*c)
    for n in range(nruns):
        thetas = np.random.uniform(-1,1,3) 
        parameters = np.array([c, x0, t0])
        a1 = a1_old * c
        a2 = a2 * c
        a3 = a3 * c
        parameters += np.array([a1, a2, a3]) * thetas

        integrating_class.c =  parameters[0]
        integrating_class.x0 =  parameters[1]
        integrating_class.sigma  = parameters[1]
        integrating_class.t0  = parameters[2]

        integrating_class.recall_collided_uncollided_classes()
        integrating_class.integrate(tfinal, npnts)
        xs, phi, phi_u = integrating_class.return_sol()
        # plt.plot(xs, phi_u, "k--")
        # plt.plot(xs, phi, "bo", mfc = 'none')
        result_list[0] = xs
        result_list[n+1] = phi

    statistics_array = np.zeros((5, npnts))
    statistics_array[0] = xs

    
    print(a1, 'a1')
    analytic_exp = integrating_class.first_mom
    analytic_var = integrating_class.second_mom
    for ix in range(npnts):
        results = stats.describe(result_list[1:, ix])
        # print('x=', result_list[0, ix], result_list[1:, ix])
        statistics_array[1,ix] = results.mean
        statistics_array[2,ix] = results.variance
        # statistics_array[2,ix] = np.mean(result_list[1:, ix]**2) - np.mean(result_list[1:, ix])**2
        statistics_array[3,ix] = results.skewness
        statistics_array[4,ix] = results.kurtosis
    # plt.plot(statistics_array[0], statistics_array[1], 'b-', label = 'sampled mean')
    # plt.plot(statistics_array[0], statistics_array[2], 'g-', label = 'sampled var')
    # plt.plot(statistics_array[0], statistics_array[3], 'y-', label = 'samp,ed skew')
    # plt.show()
    # plt.figure(2)
    # plt.hist(result_list[1:, 0])
    # plt.show()
    return statistics_array, analytic_exp, analytic_var

        




