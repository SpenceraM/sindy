import numpy as np
import matplotlib.pyplot as plt


class DynamicSystem:
    def __init__(self, x0, t0, dt, t_end, f):
        self.x = x0
        self.t = t0
        self.dt = dt
        self.t_end = t_end

    def f(self, x, t):
        pass

    def step(self):
        self.x = self.x + self.dt * self.f(self.x, self.t)
        self.t = self.t + self.dt
        return self.x

    def run(self):
        xs = []
        while self.t < self.t_end:
            xs.append(self.step())
        return np.array(xs)

    def get_derivative(self, states):  # save the trajectory and derivatives
       return np.gradient(states, axis=0)/self.dt

    @staticmethod
    def plot(self): # display in relevant space
        pass




class Lorenz(DynamicSystem):
    def __init__(self, x0, t0, dt, t_end, f, sigma, rho, beta):
        super().__init__(x0, t0, dt, t_end, f)
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
        """
        %% compute Sparse regression: sequential least squares
        Xi = Theta\dXdt;  % initial guess: Least-squares
        % lambda is our sparsification knob.
        for k=1:10
            smallinds = (abs(Xi)<lambda);      % find small coefficients
            Xi(smallinds)=0;                   % and threshold
            for ind = 1:n                      % n is state dimension
                biginds =  ~smallinds(:,ind);
                % Regress dynamics onto remaining terms to find sparse Xi
                Xi(biginds,ind) = Theta(:,biginds)\dXdt(:,ind);
            end
        end
        """
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

    def get_sparse_coefficients(self):
        if self.library is None:
            self.get_library()
        # self.coefficients = np.linalg.lstsq(self.library, self.derivatives, rcond=None)[0]
        self.coefficients = self.stls(.7)


if __name__ == '__main__':

    lorenz = Lorenz(x0=np.array([1, 1, 1]), t0=0, dt=0.01, t_end=50, f=None, sigma=10, rho=28, beta=8 / 3)
    states1 = lorenz.run()
    derivatives1 = lorenz.get_derivative(states1)
    lorenz = Lorenz(x0=np.array([-1, 2, 3]), t0=0, dt=0.01, t_end=50, f=None, sigma=10, rho=28, beta=8)
    states2 = lorenz.run()
    derivatives2 = lorenz.get_derivative(states2)
    lorenz = Lorenz(x0=np.array([-5, 7.3, -12]), t0=0, dt=0.01, t_end=50, f=None, sigma=10, rho=28, beta=8)
    states3 = lorenz.run()
    derivatives3 = lorenz.get_derivative(states3)
    lorenz = Lorenz(x0=np.array([-1, -1, -1]), t0=0, dt=0.01, t_end=50, f=None, sigma=10, rho=28, beta=8)
    states4 = lorenz.run()
    derivatives4 = lorenz.get_derivative(states4)
    lorenz = Lorenz(x0=np.array([-3, -20, 30]), t0=0, dt=0.01, t_end=50, f=None, sigma=10, rho=28, beta=8)
    states5 = lorenz.run()
    derivatives5 = lorenz.get_derivative(states5)
    lorenz = Lorenz(x0=np.array([-5.7, 7.3, -30]), t0=0, dt=0.01, t_end=50, f=None, sigma=10, rho=28, beta=8)
    states6 = lorenz.run()
    derivatives6 = lorenz.get_derivative(states6)

    states_comb = np.vstack((states1,states2,states3,states4,states5,states6))
    derivatives_comb = np.vstack((derivatives1,derivatives2,derivatives3,derivatives4,derivatives5,derivatives6))
    # Lorenz.plot(states3)
    # plt.show()

    lorenz_solver = SindySolver(states_comb, derivatives_comb, poly_order=2, threshold=0.5)
    lorenz_solver.get_library()
    lorenz_solver.get_sparse_coefficients()

    print()
