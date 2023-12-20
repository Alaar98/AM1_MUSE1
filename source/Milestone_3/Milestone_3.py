
################################################################
###   Milestone 3: Error estimation of numerical solutions   ###
################################################################

from numpy import array, zeros, arange, linspace
import matplotlib.pyplot as plt
from ODEs.Cauchy_Problem import Cauchy_problem
from ODEs.Temporal_Schemes import Euler, CN, RK4, Inverse_Euler
from Functions_Milestone_3 import Richardson_error, Convergency
from ODEs.Orbits import Kepler

# Function to evaluate errors by means of Richardson
def plot_error_comparison():
    # Define parameters for Euler, CN, RK4, and Inverse Euler
    dt = 0.01
    t = arange(0, 200, dt)
    Uo = array([1, 0, 0, 1])
    schemes = [Euler, CN, RK4, Inverse_Euler]

    # Compute Richardson error for each scheme
    plt.axis('equal')
    for scheme in schemes:
        E = Richardson_error(Kepler, t, Uo, scheme)
        plt.plot(t, E[0, :], label=scheme.__name__)

    # Graph configuration
    plt.xlabel('t')
    plt.ylabel('ERROR')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.title('Numerical error of different schemes solving Kepler')
    plt.show()


# Function to evaluate the convergence rate of different temporal schemes
def plot_convergence_rate():
    schemes = [Euler, CN, RK4]
    p = 5
    N = 2000
    t = linspace(0, 10, N)
    Uo = array([1, 0, 0, 1])

    # Compute convergence rate for Euler, CN, and RK4 methods
    for scheme in schemes:
        [log_E, log_N] = Convergency(Kepler, t, Uo, scheme, p)
        print(f"{scheme.__name__}: {log_N}, {log_E}")
        plt.plot(log_N, log_E, label=scheme.__name__)

        # Graph configuration
        plt.xlabel('log_N')
        plt.ylabel('log_E')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.title(f'Convergence rate {scheme.__name__} method solving Kepler')
        plt.show()


# Run functions
plot_error_comparison()
plot_convergence_rate()