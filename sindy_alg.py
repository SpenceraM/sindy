import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class DynamicSystem(ABC):
    def __init__(self, dt, t_end, f):
        self.x = None
        self.t = None
        self.dt = dt
        self.t_end = t_end
        self.coefficient_names = None
        self.state_names = None

    @abstractmethod
    def f(self, x, t):
        pass

    def step(self):
        self.x = self.x + self.dt * self.f(self.x, self.t)
        self.t = self.t + self.dt
        return self.x

    def run(self, x0, t0=0):
        self.x = x0
        self.t = t0
        xs = []
        while self.t < self.t_end:
            xs.append(self.step())
        return np.array(xs)

    def plot(self): # display in relevant space
        pass

    def get_derivative(self, states):  # save the trajectory and derivatives
       return np.gradient(states, axis=0, edge_order=2)/self.dt

    def print_coefficients(self, coefficients):
        if self.coefficient_names is None:
            self.get_library()
        print("Dynamic Equations:")
        for state_var in range(coefficients.shape[1]):
            eq_str = self.state_names[state_var] + " = "
            for lib_var in range(coefficients.shape[0]):
                if coefficients[lib_var,state_var] != 0:
                    eq_str += str(round(coefficients[lib_var,state_var],2)) + self.coefficient_names[lib_var] +  ' + '
                    # print(coefficients[i,j], self.coefficient_names[i], " = ", self.state_names[j])
            print(eq_str[:-2])

class Lorenz(DynamicSystem):
    def __init__(self, dt, t_end, f, sigma, rho, beta):
        super().__init__(dt, t_end, f)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.coefficient_names = ['1', 'x', 'y', 'z', 'x^2', 'xy', 'xz', 'y^2', 'yz', 'z^2']
        self.state_names = ['x', 'y', 'z']


    def f(self, x, t):
        x1 = x[0]  # x1 = x
        x2 = x[1]  # x2 = y
        x3 = x[2]  # x3 = z
        return np.array([self.sigma * (x2 - x1), x1 * (self.rho - x3) - x2, x1 * x2 - self.beta * x3])

    @staticmethod
    def plot(xs):
        N = xs.shape[0]
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2],c=np.arange(N)/N, cmap='jet', marker = '.', s=.5)


class SindySolver:
    def __init__(self, states, derivatives, poly_order, threshold):
        self.states = states
        self.derivatives = derivatives
        self.poly_order = poly_order
        self.threshold = threshold
        self.library = None
        self.coefficients = None


    def get_library(self):
        self.library = np.ones([self.states.shape[0],1])
        self.library = np.hstack((self.library, self.states))
        lib_extended = []
        col_counter = 0

        # Below is wrong. like 31 cols but should be less
        n_states = self.states.shape[1]
        if self.poly_order == 2:
            for order in range(2,self.poly_order+1):

                for i in range(self.states.shape[1]):
                    for j in range(i, self.states.shape[1]):
                        temp = self.states[:, i] * self.states[:, j]  # new entry
                        self.library = np.append(self.library, temp[...,np.newaxis],1)
                        col_counter += 1
        else:
            print("Need to add functionality for powers greater than 2 using Combinations")

    def stls(self, thresh, max_n=5, plot_flag=False): # sequential thresholded least-squares
        Xi = np.linalg.lstsq(self.library, self.derivatives, rcond=None)[0]
        for k in range(max_n):
            if plot_flag:
                plt.figure()
                plt.imshow(Xi,cmap='seismic',interpolation='none',aspect='auto',vmax=np.abs(np.max(Xi)),vmin=-np.abs(np.max(Xi)))
                plt.show(block=False)
            smallinds = np.abs(Xi) < thresh
            Xi[smallinds] = 0
            for ind in range(self.states.shape[1]):
                big_inds = ~smallinds[:, ind]
                Xi[big_inds, ind] = np.linalg.lstsq(self.library[:, big_inds], self.derivatives[:, ind], rcond=None)[0]
        return Xi

    def get_sparse_coefficients(self, thresh  = 0.5):
        if self.library is None:
            self.get_library()
        # self.coefficients = np.linalg.lstsq(self.library, self.derivatives, rcond=None)[0]
        self.coefficients = self.stls(thresh)

    @staticmethod
    def get_data(system, n_trials, initial_state_bounds, noise_std=0.1):
        states = []
        derivatives = []
        for i in range(n_trials):
            x0 = np.random.uniform(initial_state_bounds[0],initial_state_bounds[1],3)
            states.append(system.run(x0, t0=0))
            derivatives.append(system.get_derivative(states[-1]))
        states, derivatives = np.concatenate(states), np.concatenate(derivatives)

        # add noise to states and derivatives
        states = states + np.random.normal(0, noise_std, states.shape)
        derivatives = derivatives + np.random.normal(0, noise_std, derivatives.shape)
        return states, derivatives


if __name__ == '__main__':

    lorenz = Lorenz(dt=0.0005, t_end=50, f=None, sigma=10, rho=28, beta=2)
    # states = lorenz.run(np.array([1, 1, 1]), t0=0)
    # lorenz.plot(states)
    # plt.show(block=False)
    states, derivatives = SindySolver.get_data(lorenz, 30, [-40,40], noise_std=0.0)

    lorenz_solver = SindySolver(states, derivatives, poly_order=2, threshold=0.5)
    lorenz_solver.get_library()
    lorenz_solver.get_sparse_coefficients()
    lorenz.print_coefficients(lorenz_solver.coefficients)
    # print(lorenz_solver.coefficients)
    print()
