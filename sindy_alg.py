import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class DynamicSystem(ABC):
    def __init__(self, dt, t_end, f):
        self.x = None
        self.t = None
        self.dt = dt
        self.t_end = t_end

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
       return np.gradient(states, axis=0)/self.dt


class Lorenz(DynamicSystem):
    def __init__(self, dt, t_end, f, sigma, rho, beta):
        super().__init__(dt, t_end, f)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

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

    def get_library(self):
        self.library = np.ones([self.states.shape[0],1])
        self.library = np.hstack((self.library, self.states))
        lib_extended = []
        col_counter = 0

        # Below is wrong. like 31 cols but should be less
        n_states = self.states.shape[1]
        if self.poly_order > 1:
            for order in range(2,self.poly_order+1):

                for i in range(self.states.shape[1]):
                    for j in range(i, self.states.shape[1]):
                        temp = self.states[:, i] * self.states[:, j]  # new entry
                        self.library = np.append(self.library, temp[...,np.newaxis],1)
                        col_counter += 1
        else:
            print("Need to work with power greater than 2 using Combinations")

    def stls(self, thresh, max_n=5): # sequential thresholded least-squares
        Xi = np.linalg.lstsq(self.library, self.derivatives, rcond=None)[0]
        for k in range(max_n):
            plt.figure()
            plt.imshow(Xi,cmap='jet',interpolation='none',aspect='auto')
            plt.show(block=False)
            smallinds = np.abs(Xi) < thresh
            Xi[smallinds] = 0
            for ind in range(self.states.shape[1]):
                big_inds = ~smallinds[:, ind]
                Xi[big_inds, ind] = np.linalg.lstsq(self.library[:, big_inds], self.derivatives[:, ind], rcond=None)[0]
        pass

    def get_sparse_coefficients(self, thresh  = 0.5):
        if self.library is None:
            self.get_library()
        # self.coefficients = np.linalg.lstsq(self.library, self.derivatives, rcond=None)[0]
        self.coefficients = self.stls(thresh)

    @staticmethod
    def get_data(system, n_trials, system_params, initial_state_bounds):
        states = []
        derivatives = []
        for i in range(n_trials):
            x0 = np.random.uniform(initial_state_bounds[0],initial_state_bounds[1],3)
            states.append(system.run(x0, t0=0))
            derivatives.append(system.get_derivative(states[-1]))
        return np.concatenate(states), np.concatenate(derivatives)


if __name__ == '__main__':

    lorenz = Lorenz(dt=0.01, t_end=20, f=None, sigma=10, rho=28, beta=8)
    states, derivatives = SindySolver.get_data(lorenz, 50, None, [-30,30])

    lorenz_solver = SindySolver(states, derivatives, poly_order=2, threshold=0.5)
    lorenz_solver.get_library()
    lorenz_solver.get_sparse_coefficients()

    print()
