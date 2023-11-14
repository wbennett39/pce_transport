import numpy as np
import math
import scipy.integrate as integrate
from moving_mesh_transport.solver_classes.functions import Pn_scalar
from moving_mesh_transport.benchmarks.benchmark_functions import F1
from moving_mesh_transport.benchmarks.collided import collided_class
from numba import njit
from tqdm import tqdm
from .functions import integrate_quad
import quadpy
from numba import prange


def opts0(*args, **kwargs):
       return {'limit':1000000, 'epsabs':1.5e-10, 'epsrel':1.5e-10}


class make_expansion:
    def __init__(self, problem_name, a1, a2, a3, cbar, x0bar, t0bar, xlist, tfinal):

        self.problem_name = problem_name
        self.a1 = a1 * cbar
        self.a2 = a2 * cbar
        self.a3 = a3 * cbar
        self.cbar = cbar
        self.x0bar = x0bar
        self.t0bar = t0bar
        self.xlist = xlist
        self.tfinal = tfinal

    # def plane_integrand(self,)
    def get_coefficients_plane_pulse(self, N, x, t):
        # print('number of bases', N)
        coeffs = np.zeros(N+1)
        for n in range(N+1):
            integrand = lambda theta1: self.plane_IC(x, t, self.cbar + self.a1 * theta1) * Pn_scalar(int(n), theta1, -1.0, 1.0)
            coeffs[n] = 0.5*(2*n+1)*integrate.nquad(integrand, [[-1,1]])[0]
            # coeffs[n] = 0.5*(2*n+1)*self.integrator(integrand, -1, 1, n)
        return coeffs

    def get_coefficients_source(self, N, x, t):
        coeffs = np.zeros(N+1)
        for n in range(N+1):
            integrand = lambda theta1: self.collided_solution(x, t, self.cbar + self.a1 * theta1) * Pn_scalar(int(n), theta1, -1.0, 1.0)
            coeffs[n] = 0.5*(2*n+1)*integrate.nquad(integrand, [[-1,1]])[0]
            # coeffs[n] = 0.5*(2*n+1)*self.integrator(integrand, -1.0, 1.0, n)
        return coeffs

    def analytic_exp_plane_pulse(self, t):
        exp = np.zeros(self.xlist.size)
        var = np.zeros(self.xlist.size)
        for ix, xx in enumerate(self.xlist):
            integrand = lambda theta1: self.plane_IC(xx, t, self.cbar + self.a1 * theta1) 
            integrand_sq = lambda theta1: self.plane_IC(xx, t, self.cbar + self.a1 * theta1)**2 
            exp[ix] = integrate.nquad(integrand, [[-1,1]])[0]
            var[ix] = integrate.nquad(integrand_sq, [[-1,1]])[0]
        return exp, var

    # def analytic_exp(self, t):
    #     exp = np.zeros(self.xlist.size)
    #     var = np.zeros(self.xlist.size)
    #     for ix, xx in enumerate(self.xlist):
    #         integrand = lambda theta1: self.collided_solution(xx, t, self.cbar + self.a1 * theta1) 
    #         integrand_sq = lambda theta1: self.plane_IC(xx, t, self.cbar + self.a1 * theta1)**2 
    #         exp[ix] = integrate.nquad(integrand, [[-1,1]])[0]
    #         var[ix] = integrate.nquad(integrand_sq, [[-1,1]])[0]
    #     return exp, var

    def plane_IC(self, x, t, c):
        temp = integrate.nquad(F1, [[0, math.pi]], args =  (0.0, 0.0, x, t, 0, c), opts = [opts0])[0]
        return temp

    def collided_solution(self, x, t, c):
        x0 = 0.5
        t0 = 5
        sigma = 0.5
        collided_class_ob = collided_class(self.problem_name, x0, t0, sigma)
        return collided_class_ob(x, t, c)

    def get_coefficients(self, N):
        coeffs_list = np.zeros((N+2, self.xlist.size))
        coeffs_list[0] = self.xlist
        for ix, xx in enumerate(tqdm(self.xlist)):
            if self.problem_name == 'plane_IC':
                coeffs = self.get_coefficients_plane_pulse(N, self.xlist[ix], self.tfinal)
            else:
                coeffs = self.get_coefficients_source(N, np.array([self.xlist[ix]]), self.tfinal)
            
            coeffs_list[1:, ix] = coeffs

        return coeffs_list
    
    def integrator(self, integrand, a, b, n, tol = 1e-8):
        npts = 2*n+2
        err = 1
        it = 0
        res_old = 1.0
        while err >= tol:
            xs = quadpy.c1.gauss_legendre(npts).points
            ws = quadpy.c1.gauss_legendre(npts).weights
            evaluated_integrand = np.zeros(xs.size)
            for ix, xx in enumerate(xs):
                evaluated_integrand[ix] = integrand((b-a)/2*xx + (a+b)/2)
            res = integrate_quad(a, b, xs, ws, evaluated_integrand)
            it += 1
            npts = npts*2
            if it >= 2:
                err = np.abs(res - res_old)/res_old
                res_old = res
                # print(err, "npts = ", npts)
        return res


@njit
def evaluate(theta, N, coeff):
    res = 0 
    for n in range(N+1):
        res += coeff[n] * Pn_scalar(int(n), theta, -1.0, 1.0)
    return res
@njit(parallel=True)
def sampling_func(samples, N, coeff, uncollided):
    res = np.zeros(samples.size)
    for i in prange(samples.size):
        theta = samples[i,0]*2-1
        res[i] = evaluate(theta, N, coeff) + uncollided
    return res


