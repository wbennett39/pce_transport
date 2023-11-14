import sys
import matplotlib.pyplot as plt
# sys.path.append('/Users/bennett/Documents/Github/MovingMesh/moving_mesh_radiative_transfer')

from moving_mesh_transport.solver_classes.functions import normPn_scalar
from numba import njit, prange
import numpy as np
@njit 
def expansion_polynomial(coeffs, x):
    res = 0.0
    for n in prange(coeffs.size):
        res += coeffs[n] * normPn_scalar(n,x, -1.0, 1.0)
    return res

@njit 
def expansion_polynomial_squared(coeffs, x):
    res = 0.0
    for n in prange(coeffs.size):
        res += (coeffs[n] * normPn_scalar(n,x, -1.0, 1.0))**2
    return res

@njit 
def integrate_quad(a, b, xs, ws, func1_evaluated):
    # val = xs* 0 
    # for ix, xx in enumerate(xs):
    # val = np.zeros(xs.size)
    # for ix, xx in enumerate(xs):
    #     val[ix] = func1((b-a)/2*xx + (a+b)/2)
    res = (b-a)/2 * np.sum(ws * func1_evaluated)
    return res

