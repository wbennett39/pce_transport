import sys
import matplotlib.pyplot as plt
sys.path.append('/Users/bennett/Documents/Github/MovingMesh/moving_mesh_radiative_transfer')

import numpy as np
import matplotlib.pyplot as plt
from .integrator import sampling_integrator
from .pc_expansion import make_expansion, sampling_func
from .functions import expansion_polynomial, expansion_polynomial_squared
import math
import scipy.integrate as integrate
from moving_mesh_transport.plots.plot_functions.show_loglog import show_loglog
from moving_mesh_transport.plots.plot_functions.show import show
from scipy.stats import qmc
from moving_mesh_transport.benchmarks.uncollided import uncollided_class

def RMSE(a,b):
    return math.sqrt(np.mean((a-b)**2))

def skew(coeffs, phi_u, expectation, variance):
    skew = 0.0
    skew_2 = 0.0
    if variance > 0:
        std = math.sqrt(variance)
    
        skew = 0.0
        N = coeffs.size
        third_moment = 0.0
        third_moment += 3 * phi_u * np.sum(coeffs**2) / 2
        third_moment += 3 * phi_u**2 * coeffs[0] / math.sqrt(2)
        expansion_cubed = lambda theta1: expansion_polynomial(coeffs, theta1) ** 3
        third_moment += 0.5 * integrate.nquad(expansion_cubed, [[-1,1]])[0] 
        third_moment += phi_u**3
        mean = math.sqrt(2/3) * coeffs[1] * 0.5 
        if std != 0:
            skew = (third_moment - 3 * expectation * std**2 - expectation**3) / std**3

        third_moment_integrand = lambda theta1: ((expansion_polynomial(coeffs, theta1) + phi_u - expectation)/ std)**3
        skew_2 = 0.5 * integrate.nquad(third_moment_integrand, [[-1,1]])[0]


    return skew, skew_2


def test_plane_pulse(tfinal = 1.0, c = 1.0, a1 = 0.1, npnts = 100, nruns = 1, N = 15, lst = '-', clr = 'tab:blue', nsamples = 2):
    stats_from_sampler, analytic_exp2, analytic_var = sampling_integrator(tfinal, 0.0, 0.0, c, a1, 0.0, 0.0, 'plane_IC', npnts, nruns)
    xlist = stats_from_sampler[0]
    coefficient_maker = make_expansion('plane_IC', a1, 0, 0, c, 0, 0, xlist, tfinal)
    
    N_bases = np.array([1,2,3,4,5,6,7,8])
    RMSE_mean_list = np.zeros(N_bases.size)
    RMSE_var_list = np.zeros(N_bases.size)
    RMSE_skew_list = np.zeros(N_bases.size)
    res = coefficient_maker.analytic_exp_plane_pulse(tfinal)
    sampler = qmc.Sobol(d=1, scramble=False)
    samples = sampler.random_base2(m=nsamples)
    analytic_exp = 0.5*res[0] + math.exp(-tfinal)/2.0/tfinal
    # analytic_exp = np.append(analytic_exp, 0.0)
    # analytic_exp2 = np.append(analytic_exp2, 0.0)
    analytic_var2 = 0.5 * res[1] - 0.25 * res[0]**2
    

    for count, nn in enumerate(N_bases):
        print('N = ', nn)
        print(count, 'count')
        print(xlist.size, 'xlist size')
        coeffs = coefficient_maker.get_coefficients(nn)
        print(np.shape(coeffs), 'shape coeffs')
        expectation_list = coeffs[1,:] + math.exp(-tfinal)/2.0/tfinal
        RMSE_mean = RMSE(expectation_list, stats_from_sampler[1])
        expectation_list = np.append(expectation_list, 0.0)
        
        # print('RMSE mean ', RMSE_mean)
        RMSE_mean_list[count] = RMSE_mean
  
        
        var_list = np.zeros(xlist.size)
        skew_list = np.zeros(xlist.size)
        skew_list_2 = np.zeros(xlist.size)

        median = np.zeros(xlist.size)
        twentieth = np.zeros(xlist.size)
        eightieth = np.zeros(xlist.size)
        for iv in range(npnts):
            uncollided = math.exp(-tfinal)/2/tfinal
            sampled_vals = sampling_func(samples, nn, coeffs[1:,iv], uncollided)
            median[iv] = np.quantile(sampled_vals, 0.5)
            eightieth[iv] = np.quantile(sampled_vals, 0.8)
            twentieth[iv] = np.quantile(sampled_vals, 0.2)
            expectation = coeffs[1,iv] + math.exp(-tfinal)/2.0/tfinal
            # print(RMSE(expectation, analytic_exp), "RMSE exp")

            # mean = math.sqrt(2/3) * 0.5 * coeffs[2, iv]
            # var = np.sum((coeffs[1:,iv])**2)/2  + math.sqrt(2)*uncollided * (coeffs[1,iv])  + uncollided**2 - (expectation**2)
            var_integrand = lambda theta1: (expansion_polynomial(coeffs[1:, iv], theta1) + uncollided)**2
            var2 = 0.5 * integrate.nquad(var_integrand, [[-1,1]])[0] - expectation**2
            var = 0
            for i in range(2, nn+1):
                var += coeffs[i, iv]**2/(2*(i-1)+1)
            var3 = 0.5 * np.sum((coeffs[2:,iv])**2) 
            # if abs(var-var3) > 1e-15:
            #     print(abs(var-var3), 'variance diff')
            var_list[iv] = var
            # print(var, 'var')
            skew_list[iv] = skew(coeffs[1:, iv], uncollided, expectation, var)[0]
            skew_list_2[iv] = skew(coeffs[1:, iv], uncollided, expectation, var)[1]
            # if abs(skew_list[iv] - skew_list_2[iv]) > 1e-16:
            #     print('skew diff', abs(skew_list[iv] - skew_list_2[iv]) )
        

        
        plt.figure(7)
        std = np.sqrt(var_list)
        std = np.append(std, 0.0)



        ylist = coeffs[1,:] + math.exp(-tfinal)/2.0/tfinal

        xlist2 = np.append(xlist, tfinal + 1e-8)

        ylist2 = np.append(ylist, 0.0)

        print(RMSE(ylist, analytic_exp2), 'RMSE 2')
        rmse_mean = RMSE(ylist, analytic_exp2)
        rmse_var = RMSE(var_list, analytic_var2)
        
        


        legend_string = r'$\overline{c} = $' + str(c)
        plt.plot(xlist2, ylist2, label = legend_string, ls = '-', c = clr )
        plt.plot(xlist2, ylist2-std, ls = '--', c = clr )
        plt.plot(xlist2, ylist2+std, ls = '--', c = clr )
        # plt.plot(xlist, analytic_exp, 'o', c = clr, mfc = 'none')
        uncol = math.exp(-tfinal)/2.0/tfinal
        # plt.plot(xlist,analytic_exp2, '^', c = 'k', mfc = 'none')

        print(RMSE(ylist, analytic_exp), "RMSE exp")
        print(RMSE(var_list, analytic_var2), "RMSE var")
        RMSE_var_list[count] = RMSE(var_list, analytic_var2)
        plt.legend()
        plt.xlabel('x', fontsize = 16)
        plt.ylabel(r'$\phi$', fontsize = 16)
        if c == 0.25:
            if a1 == 0.1/0.25:
                show(f'plane_pulse_t={tfinal}_moments_onetenth')
            else:
                show(f'plane_pulse_t={tfinal}_moments_tenpercent')
        plt.show()


        plt.figure(23)
        legend_string = r'$\overline{c} = $' + str(c)
        median2 = np.append(median, 0.0)
        # print(median2)
        eightieth2 = np.append(eightieth, 0.0)
        twentieth2 = np.append(twentieth, 0.0)
        plt.plot(xlist2, median2, c = clr, label = legend_string)
        plt.plot(xlist2, twentieth2, ls = '--', c = clr)
        plt.plot(xlist2, eightieth2, ls = '-.', c = clr)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.legend()
        if c == 0.25:
            if a1 == 0.1/0.25:
                show(f'plane_pulse_t={tfinal}_quantile_onetenth')
            else:
                show(f'plane_pulse_t={tfinal}_quantile_tenpercent')
        plt.show()
        # plt.figure(8)
        # analytic_var = np.append(analytic_var,0)
        # analytic_var2 = np.append(analytic_var2,0)
        # plt.plot(xlist, analytic_var)
        # var_list = np.append(var_list,0)
        # plt.plot(xlist, var_list)
        # plt.plot(xlist, analytic_var2, 'o', mfc = 'none')
        # plt.show()
        # # plt.close()

        # plt.figure(1)
        # plt.plot(xlist, var_list, 'rx', label ='PCE var')
        # plt.show()
        # plt.plot(xlist, skew_list, 'r--', label ='PCE skew', mfc = 'none')
        # plt.plot(xlist, skew_list_2, 'r-.', label ='PCE skew 2', mfc = 'none')

        # RMSE_var = RMSE(var_list, stats_from_sampler[2])
        # RMSE_skew = RMSE(skew_list_2, stats_from_sampler[3])

        # print(max(abs(coeffs[-1, :])), 'last coefficient max')


        # RMSE_var_list[count] = RMSE_var
        # RMSE_skew_list[count] = RMSE_skew
    if a1 == 0.1 * c:
        mkr = '-o'
    elif a1 == 0.3 * c:
        mkr = '-^'
    elif a1 == 0.5 * c:
        mkr = '-s'
    else:
        mkr = '-*' 

    print(RMSE_mean_list, 'mean')
    print(RMSE_var_list, 'var')
    print(RMSE_skew_list, 'skew')
    # plt.figure(4)
    # plt.loglog(N_bases, RMSE_mean_list, mkr, mfc = 'none', label = f'a_1 = {a1}')
    # # plt.legend()
    # show_loglog(f'tpl_mean_c={c}', 1.5, 12)
    # # plt.show()

    # plt.figure(5)
    # plt.loglog(N_bases, RMSE_var_list,  mkr, mfc = 'none', label = f'a_1 = {a1}')
    # # plt.legend()
    # # show_loglog(f'tpl_var_c={c}', 1.5, 12)
    # # plt.show()
    

    # plt.figure(6)
    # plt.loglog(N_bases, RMSE_skew_list, mkr, mfc = 'none', label = f'a_1 = {a1}')
    # plt.legend()
    # show_loglog(f'tpl_skew_c={c}', 1.5, 12)
    # plt.show()
    # plt.show()

    # plt.figure(1)
    # plt.plot(xlist, expectation_list, 'o', label = 'PCE expectation')
    # plt.plot(xlist, stats_from_sampler[1], 'k-', label = 'sampled')
    # plt.legend()
    # plt.show()

    # plt.figure(2)
    # plt.plot(xlist, var_list, 'o', label = 'PCE var')
    # plt.plot(xlist, stats_from_sampler[2], 'k-', label = 'sampled')
    # plt.legend()
    # plt.show()

    # plt.figure(3)
    # plt.plot(xlist, skew_list, 'o', label = 'PCE SKEW')
    # plt.plot(xlist, stats_from_sampler[3], 'k-', label = 'sampled')
    # plt.legend()
    # plt.show()

    # return RMSE_mean_list, RMSE_var_list, RMSE_skew_list, coeffs, xlist, ylist, std
    plt.figure(9)
    plt.plot(N_bases, (np.sqrt(RMSE_var_list)), '-o')
    plt.ylabel('RMSE', fontsize= 16)
    plt.xlabel('j', fontsize= 16)
    plt.yscale('log')
    show_loglog(f'convergence_t={tfinal}_c={c}_a1={a1}', 0.0001, 8.5)
    return RMSE_var_list




        # plt.plot(xlist, coeffs[1,:]/math.sqrt(2) + math.exp(-tfinal)/2.0/tfinal + np.sqrt(var_list), label = "plus var")
        # plt.plot(xlist, coeffs[1,:]/math.sqrt(2) + math.exp(-tfinal)/2.0/tfinal - np.sqrt(var_list), label = 'minus var')
        # plt.legend()




def test_source(tfinal = 1.0, c = 0.25, a1 = 0.1, problem_name = 'square_source', x0 = 0.5, t0 = 5, npnts = 20, nruns = 1, N = 15, lst = '-', clr = 'tab:blue', nsamples = 2):
    # stats_from_sampler, analytic_exp2, analytic_var = sampling_integrator(tfinal, 0.0, 0.0, c, a1, 0.0, 0.0, 'plane_IC', npnts, nruns)
    # xlist = stats_from_sampler[0]
    xlist = np.linspace(0, x0 + tfinal, npnts)

    coefficient_maker = make_expansion(problem_name, a1, 0, 0, c, 0, 0, xlist, tfinal)
    
    N_bases = np.array([4])
    RMSE_mean_list = np.zeros(N_bases.size)
    RMSE_var_list = np.zeros(N_bases.size)
    RMSE_skew_list = np.zeros(N_bases.size)
    sampler = qmc.Sobol(d=1, scramble=False)
    samples = sampler.random_base2(m=nsamples)
    uncollided_ob = uncollided_class(problem_name, x0, t0)



    for count, nn in enumerate(N_bases):
        print('N = ', nn)
        print(xlist.size, 'xlist size')
        coeffs = coefficient_maker.get_coefficients(nn)
        print('coefficients loaded')

        
        # RMSE_mean = RMSE(expectation_list, stats_from_sampler[1])
        # expectation_list = np.append(expectation_list, 0.0)
        
        # print('RMSE mean ', RMSE_mean)
        # RMSE_mean_list[count] = RMSE_mean
  
        
        var_list = np.zeros(xlist.size)
        skew_list = np.zeros(xlist.size)
        skew_list_2 = np.zeros(xlist.size)

        median = np.zeros(xlist.size)
        twentieth = np.zeros(xlist.size)
        eightieth = np.zeros(xlist.size)
        
        uncollided_sol = uncollided_ob(xlist, tfinal)
        expectation_list = coeffs[1,:] + uncollided_sol

        for iv in range(npnts):
            print(iv)
            sampled_vals = sampling_func(samples, nn, coeffs[1:,iv], uncollided_sol[iv])
            median[iv] = np.quantile(sampled_vals, 0.5)
            eightieth[iv] = np.quantile(sampled_vals, 0.8)
            twentieth[iv] = np.quantile(sampled_vals, 0.2)
            expectation = coeffs[1,iv] + uncollided_sol[iv]
            # print(RMSE(expectation, analytic_exp), "RMSE exp")

            # mean = math.sqrt(2/3) * 0.5 * coeffs[2, iv]
            # var = np.sum((coeffs[1:,iv])**2)/2  + math.sqrt(2)*uncollided * (coeffs[1,iv])  + uncollided**2 - (expectation**2)
            var_integrand = lambda theta1: (expansion_polynomial(coeffs[1:, iv], theta1) + uncollided_sol[iv])**2
            var2 = 0.5 * integrate.nquad(var_integrand, [[-1,1]])[0] - expectation**2
            var = 0
            for i in range(2, nn+1):
                var += coeffs[i, iv]**2/(2*(i-1)+1)
            var3 = 0.5 * np.sum((coeffs[2:,iv])**2) 
            # if abs(var-var3) > 1e-15:
            #     print(abs(var-var3), 'variance diff')
            var_list[iv] = var
            # print(var, 'var')
            skew_list[iv] = skew(coeffs[1:, iv], uncollided_sol[iv], expectation, var)[0]
            skew_list_2[iv] = skew(coeffs[1:, iv], uncollided_sol[iv], expectation, var)[1]
            # if abs(skew_list[iv] - skew_list_2[iv]) > 1e-16:
            #     print('skew diff', abs(skew_list[iv] - skew_list_2[iv]) )
    
        
        plt.figure(7)
        std = np.sqrt(var_list)

        ylist = coeffs[1,:] + uncollided_sol[iv]

        legend_string = r'$\overline{c} = $' + str(c)
        plt.plot(xlist, ylist, label = legend_string, ls = '-', c = clr )
        plt.plot(xlist, ylist-std, ls = '--', c = clr )
        plt.plot(xlist, ylist+std, ls = '--', c = clr )
        # plt.plot(xlist, analytic_exp, 'o', c = clr, mfc = 'none')

        # plt.plot(xlist,analytic_exp2, '^', c = 'k', mfc = 'none')
        plt.legend()
        plt.xlabel('x', fontsize = 16)
        plt.ylabel(r'$\phi$', fontsize = 16)
        if c == 0.25:
            if a1 == 0.1/0.25:
                show(f'{problem_name}_t={tfinal}_moments_onetenth')
            else:
                show(f'{problem_name}_t={tfinal}_moments_tenpercent')
        plt.show()


        plt.figure(23)
        legend_string = r'$\overline{c} = $' + str(c)

        # print(median2)

        plt.plot(xlist, median, c = clr, label = legend_string)
        plt.plot(xlist, twentieth, ls = '--', c = clr)
        plt.plot(xlist, eightieth, ls = '-.', c = clr)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.legend()
        if c == 0.25:
            if a1 == 0.1/0.25:
                show(f'{problem_name}_t={tfinal}_quantile_onetenth')
            else:
                show(f'{problem_name}_t={tfinal}_quantile_tenpercent')
        plt.show()

        # RMSE_skew_list[count] = RMSE_skew
    if a1 == 0.1 * c:
        mkr = '-o'
    elif a1 == 0.3 * c:
        mkr = '-^'
    elif a1 == 0.5 * c:
        mkr = '-s'
    else:
        mkr = '-*' 


   
    return RMSE_var_list