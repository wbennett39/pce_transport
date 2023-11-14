import numpy as np
import matplotlib.pyplot as plt
from .transport_pce import pce_transport
from matplotlib.animation import FuncAnimation
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from moving_mesh_transport.plots.plot_functions.show import show


problem_name = 'plane_IC'

def plot_video(problem_name = problem_name):


    nt = 30
    c = 0.5
    tlist = np.linspace(0.1, 10, nt)
    pce_ob = pce_transport(tlist = tlist, problem_name = problem_name, a1 = 0.1, c = c)
    pce_ob.get_coefficients()
    pce_ob.mean_var_quantiles()

    pce_ob_nominal = pce_transport(tlist = tlist, problem_name = problem_name, a1 = 1e-10, c = c)
    pce_ob_nominal.get_coefficients()
    pce_ob_nominal.mean_var_quantiles()
    energy = np.zeros(tlist.size)
    energy_nom = np.zeros(tlist.size)
    energy_median = np.zeros(tlist.size)

#   fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Horizontally stacked subplots')
    for it, tt in enumerate(tlist):
        dt = (tlist[-1]-tlist[0])/nt
        xs = np.append(pce_ob.xlist[it],pce_ob.xlist[it,-1] + 1e-12 ) 
        phi_exp = np.append(pce_ob.expectation_mat[it],0.0)
        phi_interpolated = interp1d(xs, phi_exp)
        integral_of_phi = integrate.quad(phi_interpolated, xs[0], xs[-1])[0]
        energy[it] = integral_of_phi

        phi_exp_nominal = np.append(pce_ob_nominal.expectation_mat[it], 0.0)
        phi_interpolated_nom = interp1d(xs, phi_exp_nominal)
        integral_of_phi_nominal = integrate.quad(phi_interpolated_nom, xs[0], xs[-1])[0]
        energy_nom[it] = integral_of_phi_nominal

        phi_var= pce_ob.var_mat[it]
        phi_std = np.sqrt(phi_var)
        plot_edges = np.linspace(-tt, tt, 9)

        phi_median = np.append(pce_ob.q3_list[it],0)
        phi_interpolated_median = interp1d(xs, phi_median)
        integral_of_phi_median = integrate.quad(phi_interpolated_median, xs[0], xs[-1])[0]
        energy_median[it] = integral_of_phi_median



   
#         plt.figure(1)
#         plt.title(f't = %.1f' % tt)
#         plt.plot(xs, phi_exp_nominal, 'k-', label = r'$\phi$')
#         plt.plot(-xs, phi_exp_nominal, 'k-')
# #        plt.plot(plot_edges, plot_edges*0, '|k',markersize=20)
#         plt.xlabel('x', size = 16)

      
         ## this code below plots the nominal pm 1 std and the integral of the energy
         
         
#        ax1.cla()
#        ax2.cla()
#        ax1.plot(xs, phi_exp, 'k-', label = r'E[$\phi$]')
#        ax1.plot(xs, phi_exp_nominal, 'k:', label = r'$nominal \phi$')
#        ax1.plot(xs[0:-1], phi_exp[0:-1] + phi_std, 'k--', label = r'$E[$\phi$]$ $\pm\:1\sigma$')
#        ax1.plot(xs[0:-1], phi_exp[0:-1] - phi_std, 'k--')
#        ax1.plot(-xs, phi_exp, 'k-')
#        ax1.plot(-xs, phi_exp_nominal, 'k:')
#        ax1.plot(-xs[0:-1], phi_exp[0:-1] + phi_std, 'k--')
#        ax1.plot(-xs[0:-1], phi_exp[0:-1] - phi_std, 'k--')
#        ax1.set_xlim([-7, 7])
#        ax1.set_ylim([0, 2.4])
#        ax1.set_xlabel('x')
#        ax1.set_ylabel(r'$\phi$')
#        ax1.legend()
#        # plt.pause(dt)
#        # plt.savefig(f"\{problem_name}_\{problem_name}_{it}.jpg")
#
#
#        # plt.figure(2)
#        # plt.clf()
#        ax2.plot(tlist, energy, 'k-', label = r'integral of $E[\phi]$')
#        ax2.set_xlabel('t')
#        ax2.set_ylabel(r'$\int dx\phi$')
#        ax2.legend()
#         ax1.set_xlim([0, 5])
#        ax2.set_ylim([0,20])
#        # plt.legend()
        # plt.pause(dt)
        # plt.xlim(-8.5, 8.5)
        # plt.ylim(0, 0.8)
        # plt.legend()
        # plt.savefig(f"{problem_name}_{it}.jpg")
        # plt.clf()


    plt.show()
    plt.figure(2)
    plt.plot(tlist, 2*energy, label ='expected')
    plt.plot(tlist, np.exp(np.array(tlist)*(c-1)), 'bo', mfc = 'none',label = 'analytic')
    plt.plot(tlist, 2*energy_nom, label ='nominal')
    plt.plot(tlist, 2*energy_median, 'r^', mfc = 'none', label ='median')
    plt.legend()
    plt.show()
    


def plot_energy(tt = 1, cmax = 1.5, a1 = 0.1):
    problem_name = 'plane_IC'
    nc = 50
    t = np.array([tt])
    clist = np.linspace(0, cmax, nc)
    energy = np.zeros(clist.size)
    energy_nom = np.zeros(clist.size)
    energy_median = np.zeros(clist.size)
    energy_std_p = np.zeros(clist.size)
    energy_std_m = np.zeros(clist.size)

    ZERO = 0.0
    
    for ic, cc in enumerate(clist):

        pce_ob = pce_transport(tlist = t , problem_name = problem_name, a1 = a1, c = cc)
        pce_ob.get_coefficients()
        pce_ob.mean_var_quantiles()
        phi_exp = np.append(pce_ob.expectation_mat[0],0.0)
        xs = np.append(pce_ob.xlist[0],pce_ob.xlist[0,-1] + 1e-12 ) 
        phi_interpolated = interp1d(xs, phi_exp)
        integral_of_phi = integrate.quad(phi_interpolated, xs[0], xs[-1])[0]
        energy[ic] = integral_of_phi

        # pce_ob_nominal = pce_transport(tlist = t, problem_name = problem_name, a1 = ZERO, c = cc)
        # pce_ob_nominal.get_coefficients()
        # pce_ob_nominal.mean_var_quantiles() 

        # phi_exp_nominal = np.append(pce_ob_nominal.expectation_mat[0], 0.0)
        # phi_interpolated_nom = interp1d(xs, phi_exp_nominal)
        # integral_of_phi_nominal = integrate.quad(phi_interpolated_nom, xs[0], xs[-1])[0]
        # energy_nom[ic] = integral_of_phi_nominal

        phi_var= pce_ob.var_mat[0]
        phi_std = np.sqrt(phi_var)
        plot_edges = np.linspace(-t, t, 9)

        phi_median = np.append(pce_ob.q2_list[0],0)
        phi_interpolated_median = interp1d(xs, phi_median)
        integral_of_phi_median = integrate.quad(phi_interpolated_median, xs[0], xs[-1])[0]
        energy_median[ic] = integral_of_phi_median

        phi_var= np.append(pce_ob.var_mat[0],0)
        phi_std = np.sqrt(phi_var)
        phi_interpolated_std_p = interp1d(xs, phi_std + phi_interpolated(xs))
        phi_interpolated_std_m = interp1d(xs, -phi_std + phi_interpolated(xs))
        integral_of_phi_std_p = integrate.quad(phi_interpolated_std_p, xs[0], xs[-1])[0]
        integral_of_phi_std_m = integrate.quad(phi_interpolated_std_m, xs[0], xs[-1])[0]
        energy_std_p[ic] = integral_of_phi_std_p
        energy_std_m[ic] = integral_of_phi_std_m
        plt.show()
    plt.figure(2)
    plt.ion()
    carray = np.array(clist)
    plt.plot(clist, 2*energy, 's', label ='expected', mfc = 'none')
    plt.plot(clist, np.exp(t[0]*(carray-1)), '-', mfc = 'none',label = 'nominal', c= 'k')
    # plt.plot(clist, 2*energy_nom, label ='nominal')
    plt.plot(clist, 2*energy_median, '^', mfc = 'none', label ='median', c = 'tab:orange')
    plt.plot(clist, 2*energy_std_p, '--', mfc = 'none', label =r'$\pm 1 \: \sigma$', c = 'tab:green')
    plt.plot(clist, 2*energy_std_m, '--', mfc = 'none', c = 'tab:green')
    # plt.plot()


    a1s = a1*carray
    a1 = 1
    tt = t[0]
    # plt.plot(clist, (np.exp((-1 + carray - a1*carray)*t)*(-1 + np.exp(2*a1*carray*tt)))/(2.*a1*carray*tt+1e-12), '--g', label = 'analytic exp')
    plt.legend()
    plt.ylim(0,1.5)
    plt.xlabel(r'$\overline{c}$', fontsize=16)
    plt.ylabel(r'$\overline{\phi}$', fontsize=16)
    show('mass_vs_c')
    plt.show()

    plt.figure(3)
    plt.plot(clist, abs(2*energy_median-np.exp(t[0]*(carray-1))))


def plot_quantiles(t=1, a1 = 0.1, cc = 1.0):

    pce_ob = pce_transport(tlist = np.array([t]) , problem_name = problem_name, a1 = a1, c = cc)
    pce_ob.get_coefficients()
    pce_ob.mean_var_quantiles()
    xs = np.append(pce_ob.xlist[0],pce_ob.xlist[0,-1] + 1e-12 ) 
    phi_median = np.append(pce_ob.q2_list[0],0)
    phi_80 = np.append(pce_ob.q3_list[0],0)

    nom_ob  = pce_transport(tlist = np.array([t]) , problem_name = problem_name, a1 = 0, c = cc)
    nom_ob.get_coefficients()
    nom_ob.mean_var_quantiles()
    phi_nominal = np.append(nom_ob.expectation_mat[0],0.0)


    nom_ob2  = pce_transport(tlist = np.array([t]) , problem_name = problem_name, a1 = 0, c = cc + a1*cc*0.6)
    nom_ob2.get_coefficients()
    nom_ob2.mean_var_quantiles()
    phi_nominal2 = np.append(nom_ob2.expectation_mat[0],0.0)


    plt.plot(xs, phi_nominal, 'b-', label = 'nominal')
    plt.plot(xs, phi_80, 'go', label = '80th', mfc = 'none')
    plt.plot(xs, phi_median, 'ro',mfc='none', label = 'median')
    plt.plot(xs, phi_nominal2,'k-', label = r'nominal with $c = \overline{c} + a_1 \omega_1 0.6$')
    plt.legend()
    plt.show()


