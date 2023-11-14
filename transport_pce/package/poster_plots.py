import numpy as np
import matplotlib.pyplot as plt
from .transport_pce import pce_transport
from matplotlib.animation import FuncAnimation
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from moving_mesh_transport.plots.plot_functions.show import show
from moving_mesh_transport.plots.plot_functions.show_loglog import show_loglog
import time




def results_getter(problem_name, a, M):
    print(problem_name)
    q1 = 0.20
    q2 = 0.5
    q3 = 0.80
    tlist = np.array([1.0, 5.0])
    clist = np.array([1.0, 0.75, 0.5, 0.25])
    npts = 100
    clr_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    t1 = time.perf_counter()
    coeffs_center_t1 = np.zeros((4, M+1))
    # for it, t in enumerate(tlist):
    for ic, c in enumerate(clist):
        pce_ob = pce_transport(tlist = tlist, problem_name = problem_name, a1 = a, c = c, npts = npts, Ms =[M], msamp = 12)
        pce_ob.get_coefficients()
        pce_ob.mean_var_quantiles(q1, q2, q3)
        coeffs_center_t1[ic,:] = pce_ob.coefficient_mat[0, 1:, int(npts/2)]
        for it, t in enumerate(tlist):
            xs = np.append(pce_ob.xlist[it],pce_ob.xlist[it,-1] + 1e-12 ) 
            phi_exp = np.append(pce_ob.expectation_mat[it],0.0)
            phi_median = np.append(pce_ob.q2_list[it],0.0)
            phi_20 = np.append(pce_ob.q1_list[it],0.0)
            phi_80 = np.append(pce_ob.q3_list[it],0.0)
            phi_var= pce_ob.var_mat[it]
            phi_std = np.sqrt(phi_var)
            phi_std= np.append(phi_std, 0.0)
            

            plotter(xs, phi_exp, phi_std,  phi_median, phi_20, phi_80, c, t, 1, a, problem_name, True, clr = clr_list[ic])
            plotter(xs, phi_exp, phi_std,  phi_median, phi_20, phi_80, c, t, 2, a, problem_name, False, clr = clr_list[ic])
            convergence_plotter(pce_ob.Ms[0], pce_ob.coefficient_mat[it, 1:, int(npts/2)], c, a, t, 3, problem_name, clr_list[ic])
            
    plt.show()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
            
    t2 = time.perf_counter()

    print("###################################")
    print('time', t2-t1)
    print("###################################")
    return M, coeffs_center_t1
        
       




# pce_ob_nominal = pce_transport(tlist = tlist, problem_name = problem_name, a1 = 0.0, c = 1.1, Ms = [0], msamp = 2)
# pce_ob_nominal.get_coefficients()
# pce_ob_nominal.mean_var_quantiles()
def convergence_plotter(M, coefficients, c, a1, t, fign, problem_name, clr = 'tab:blue'):
    problem_names = ['plane_IC', 'square_source', 'gaussian_source', 'line_source']
    fig_number = problem_names.index(problem_name) + 8 * (t == 1.0) + 16 * (t == 5.0)
    
    plt.figure(fig_number)
    plt.ion()
    x_axis = np.linspace(0, M, int(M+1))
    plt.semilogy(x_axis, np.abs(coefficients), '-o', c = clr, label = r'$\overline{c} = $' + str(c))
    plt.xlabel(r'$j$')
    plt.ylabel(r'$|a_j|$')
    plt.legend()
    # show_loglog(f'poster/poster_plot_{problem_name}__t={t}_convergence_a1={a1}', 0.0, M+1+0.5, True, [0,1,2])
    plt.xticks(np.arange(0, M+1, step = 1))
    SMALL_SIZE = 12
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.savefig(f'poster/poster_plot_{problem_name}__t={t}_convergence_a1={a1}.pdf')
    plt.ylim(10e-13, 1)
    

    

def plotter(xlist, expectation, std, median, p1, p2, c, t, fign,a1,  problem_name, moments = True, clr = 'tab:blue'):
    problem_names = ['plane_IC', 'square_source', 'gaussian_source', 'line_source']
    fig_number = problem_names.index(problem_name) + 4 * (t==5.0)
    if moments == False:
        fig_number = -1*fig_number
    ax = plt.figure(fig_number)
    plt.ion()
    
    if moments == True:
        legend_string = r'$\overline{c} = $' + str(c)
        # legend_string2 = r'$\overline{c} = $' + str(c)
        if c == 1.0:
        # Create another legend for the second line.
            line2 = plt.plot(np.zeros(1), np.zeros(1), label = r'$E[\phi]$' , ls = '-', c = 'k' )
            line3 = plt.plot(np.zeros(1), np.zeros(1), label = r'$E[\phi]\pm 1\sigma$' , ls = '--', c = 'k' )
        line1 = plt.plot(xlist, expectation, label = legend_string, ls = '-', c = clr )
        plt.plot(xlist, expectation-std, ls = '--', c = clr )
        plt.plot(xlist, expectation+std, ls = '--', c = clr )
        plt.plot(-xlist, expectation, ls = '-', c = clr )
        plt.plot(-xlist, expectation-std, ls = '--', c = clr )
        plt.plot(-xlist, expectation+std, ls = '--', c = clr )

        # Create a legend for the first line.
        # first_legend = plt.legend(handles=[line1], loc='upper right')

        # Add the legend manually to the current Axes.
        # ax = plt.gca().add_artist(first_legend)

        
            # plt.legend(handles=[line2, line3], loc='upper left')
        if problem_name != 'line_source':
            plt.xlabel('x', fontsize = 16)
        else:
            plt.xlabel('r', fontsize = 16)
            plt.xlim(0, xlist[-1])
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.legend(prop={'size':8})
        show(f'poster/poster_plot_{problem_name}_t={t}_moments_a1={a1}')

    else:
        legend_string = r'$\overline{c} = $' + str(c)
        if c == 1.0:
        # Create another legend for the second line.
            line2 = plt.plot(np.zeros(1), np.zeros(1), label = r'$50^{\mathrm{th}}$ percentile' , ls = '-', c = 'k' )
            line3 = plt.plot(np.zeros(1), np.zeros(1), label = r'$20^{\mathrm{th}}$ percentile' , ls = '--', c = 'k' )
            line4 = plt.plot(np.zeros(1), np.zeros(1), label = r'$80^{\mathrm{th}}$ percentile' , ls = '-.', c = 'k' )
        line1 = plt.plot(xlist, median, label = legend_string, ls = '-', c = clr )
        line1 = plt.plot(-xlist, median, ls = '-', c = clr )
        plt.plot(xlist, p2, ls = '--', c = clr )
        plt.plot(xlist, p1, ls = '-.', c = clr )
        plt.plot(-xlist, p2, ls = '--', c = clr )
        plt.plot(-xlist, p1, ls = '-.', c = clr)

       

        if problem_name != 'line_source':
            plt.xlabel('x', fontsize = 16)
        else:
            plt.xlabel('r', fontsize = 16)
            plt.xlim(0, xlist[-1])
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.legend(prop={'size':8})

        show(f'poster/poster_plot_{problem_name}__t={t}_percentiles_a1={a1}')






def make_plots(plane = True, square = True, gaussian = True, line = True):
    M = 6 

    coeff_list = np.zeros((4, 4, M+1))

    if plane == True:
        M, coeff_list[0] = results_getter('plane_IC', 0.1, M)
    if square == True:
        M, coeff_list[1] = results_getter('square_source', 0.1, M )
    if gaussian == True:
        M, coeff_list[2] =  results_getter('gaussian_source', 0.1, M)
    if line == True:
        M, coeff_list[3] = results_getter('line_source', 0.1, M)
    all_converge_plot(M, coeff_list, 1.0)
    all_converge_plot(M, coeff_list, 0.75)
    all_converge_plot(M, coeff_list, 0.5)
    all_converge_plot(M, coeff_list, 0.25)

# make_plots(plane = True, square = False, line = False, gaussian = False)

def all_converge_plot(M, coeff, c):
    t=1
    a1 = 0.1
    plt.figure(5)
    plt.ion()
    x_axis = np.linspace(0, M, int(M+1))
    marker_list = ['-o', '-s', '-*', '-^']
    source_list = ['plane pulse', 'square source', 'Gaussian source', 'line pulse']
    clr_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    if c == 1.0:
        index = 0
    elif c == 0.75:
        index = 1
    elif c == 0.5:
        index = 2
    elif c == 0.25:
        index = 3

    for n in range(4):
        plt.semilogy(x_axis, np.abs(coeff[n,index,:]), marker_list[n], c = clr_list[index], label = source_list[n], mfc = 'none')
        # plt.semilogy(x_axis, np.abs(coeff[n,1,:]), marker_list[n], c = 'tab:orange', mfc = 'none')
        # plt.semilogy(x_axis, np.abs(coeff[n,2,:]), marker_list[n], c = 'tab:green', mfc = 'none')
        # plt.semilogy(x_axis, np.abs(coeff[n,3,:]), marker_list[n], c = 'tab:red', mfc = 'none')

    plt.legend()
    # show_loglog(f'poster/poster_plot_{problem_name}__t={t}_convergence_a1={a1}', 0.0, M+1+0.5, True, [0,1,2])
    plt.xticks(np.arange(0, M+1, step = 1))
    plt.xlabel(r'$j$')
    plt.ylabel(r'$|a_j|$')
    SMALL_SIZE = 12
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)


    plt.show()
    plt.savefig(f'poster/poster_plot_t={t}_convergence_a1={a1}_c={c}_all.pdf')
    plt.close()
    # plt.ylim(10e-13, 1)



#make_plots(plane = True, square = True, gaussian = True, line = True)
