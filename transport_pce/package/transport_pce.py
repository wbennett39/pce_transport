import numpy as np
from .pc_expansion import make_expansion, sampling_func
from moving_mesh_transport.benchmarks.uncollided import uncollided_class
from tqdm import tqdm
from .pc_expansion import make_expansion, sampling_func
from scipy.stats import qmc
import h5py



class pce_transport:
    def __init__(self, tlist = [1.0, 5.0], Ms = [6], problem_name = 'plane_IC', c = 1.0, a1 = 0.1, xspan = [4.0, 5.0],  npts = 100, x0 = 1e-16, t0 = 5.0, msamp = 16):
        self.tlist = np.array(tlist)
        if len(Ms) == 1 and len(tlist) > 1:
            self.Ms = np.ones(len(tlist))*Ms[0]
        else:
            self.Ms = np.array(Ms)
        self.problem_name = problem_name
        self.c = c
        self.a1 = a1
        
        self.coefficient_mat = np.zeros((self.tlist.size, int(np.max(self.Ms + 2)), npts))
        self.npts = npts
        self.xlist = np.zeros((self.tlist.size, npts))
        self.uncollided_sol_mat = np.zeros((self.tlist.size, npts ))
        self.expectation_mat = np.zeros((self.tlist.size, npts))
        self.var_mat = np.zeros((self.tlist.size, npts))
        self.x0 = x0
        self.xspan = xspan
        self.t0 = t0
        self.evaluate_points()
        self.msamp = msamp
        self.q1_list = np.zeros((self.tlist.size, npts))
        self.q2_list = np.zeros((self.tlist.size, npts))
        self.q3_list = np.zeros((self.tlist.size, npts))
        self.get_uncollided_sol()

    def evaluate_points(self):
            for it, t in enumerate(self.tlist):
                if self.problem_name in ['plane_IC', 'square_IC', 'square_source', 'line_source']:
                    self.xlist[it] = np.linspace(0, t + self.x0,  self.npts)
                else:
                    self.xlist[it] = np.linspace(0, self.xspan[it] , self.npts )

    def get_coefficients(self):
        print('--- calculating expansion coefficients ---')
        f = h5py.File(f'{self.problem_name}_coef.hdf5', 'r+')
        for it, tt in enumerate(self.tlist):
            if not f.__contains__(f"t={self.tlist[it]}_M={self.Ms[0]}_{self.npts}_points_c={self.c}_a1={self.a1}"):
                f.close()
                for it, tt in enumerate((self.tlist)):
                    coefficient_maker = make_expansion(self.problem_name, self.a1, 0, 0, self.c, 0, 0, self.xlist[it], tt)
                    self.coefficient_mat[it] = coefficient_maker.get_coefficients(int(self.Ms[it]))
                self.save()
        else:
            f.close()
            self.load()

    def get_uncollided_sol(self):
        for it, tt in enumerate(self.tlist):
            uncollided_ob = uncollided_class(self.problem_name, self.x0, tt)
            uncollided_sol = uncollided_ob(self.xlist[it], tt)
            self.uncollided_sol_mat[it] = uncollided_sol

    def make_expectation(self):
        for it, tt in enumerate(self.tlist):
            self.expectation_mat[it] = self.uncollided_sol_mat[it] + self.coefficient_mat[it, 1, :]
    
    def make_var(self):
        for it, tt in enumerate(self.tlist):
            for j in range(2, int(self.Ms[it]+1)):
                self.var_mat[it] += self.coefficient_mat[it, j, :]**2/(2*(j-1)+1)


    def get_quantiles(self, q1 = 0.2, q2 = 0.5, q3 = 0.8):
            sampler = qmc.Sobol(d=1, scramble=False)
            samples = sampler.random_base2(m=self.msamp)
            print('--- sampling expansions ---')
            for it, tt in enumerate(tqdm(self.tlist)):
                for iv in range(self.xlist[it].size):
                    sampled_vals = sampling_func(samples, self.Ms[it], self.coefficient_mat[it, 1:,iv], self.uncollided_sol_mat[it,iv])
                    self.q1_list[it, iv] = np.quantile(sampled_vals, q1, method = 'inverted_cdf')
                    self.q2_list[it, iv] = np.quantile(sampled_vals, q2, method = 'inverted_cdf')
                    self.q3_list[it, iv] = np.quantile(sampled_vals, q3, method = 'inverted_cdf')

    def mean_var_quantiles(self, q1 = 0.2, q2 = 0.5, q3 = 0.8):
        print("--- calculating expectation ---")
        self.make_expectation()
        print("--- calculating variance ---")
        self.make_var()
        print("--- sampling for quantiles ---")
        self.get_quantiles(q1, q2, q3)
        
    def save(self):
        print('--- saving coefficients ---')
        f = h5py.File(f'{self.problem_name}_coef.hdf5', 'r+')
        for it in range(len(self.tlist)):
            dset3 = f.create_dataset(f"t={self.tlist[it]}_M={self.Ms[0]}_{self.npts}_points_c={self.c}_a1={self.a1}", data = self.coefficient_mat[it])
        f.close()
    
    def load(self):
        print('loading coefficients')
        f = h5py.File(f'{self.problem_name}_coef.hdf5', 'r+')
        for it in range(len(self.tlist)):
            self.coefficient_mat[it] = f[f"t={self.tlist[it]}_M={self.Ms[0]}_{self.npts}_points_c={self.c}_a1={self.a1}"][:,:]
        # print(self.coefficient_mat)
        f.close()


                        

    



    
