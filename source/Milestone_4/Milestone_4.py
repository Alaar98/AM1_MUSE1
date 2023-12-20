
##########################################################################
###   Milestone 4: Linear problems and regions of absolute stability   ###
##########################################################################

from numpy import array, zeros, arange, float64, linspace, linalg, real, imag
import matplotlib.pyplot as plt
from ODEs.Cauchy_Problem import Cauchy_problem
from ODEs.Temporal_Schemes import Euler, CN, RK4, Inverse_Euler, LF
from ODEs.Oscillator import Oscillator
from Functions_Milestone_4 import Stability_region

def Cauchy_problem_LP(F, t, Uo, temporal_scheme):
    dt = 0.01
    Nv = len(Uo)  # number of rows needed
    N = len(t) - 1  # number of columns needed
    U = zeros((Nv, N + 1), dtype=float64)
    U[:, 0] = Uo
    U[:,1] = U[:,0] + dt*F(U[:,0], t[0])

    for i in range(1,N):
        U[:, i + 1] = temporal_scheme(U[:, i], U[:, i-1], t[i], t[i+1]-t[i], F)
        
    x = U[0, :]  # Collect x values or values of the 1st row
    y = U[1, :]  # Collect y values or values of the 2nd row

    return U, x, y

# Function to integrate the linear oscillator with different methods
def harmonic_oscillator_integration():
    # Define initial conditions and time parameters
    dt = 0.01
    t = arange(0, 200, dt)
    Uo = array([1, 0])

    # Compute Cauchy problems for different methods
    F = Oscillator

    # Euler method
    temporal_scheme = Euler
    U_euler, x_euler, y_euler = Cauchy_problem(F, t, Uo, temporal_scheme)

    # Inverse Euler method
    temporal_scheme = Inverse_Euler
    U_inv_euler, x_inv_euler, y_inv_euler = Cauchy_problem(F, t, Uo, temporal_scheme)

    # CN method
    temporal_scheme = CN
    U_cn, x_cn, y_cn = Cauchy_problem(F, t, Uo, temporal_scheme)

    # RK4 method
    temporal_scheme = RK4
    U_rk4, x_rk4, y_rk4 = Cauchy_problem(F, t, Uo, temporal_scheme)

    # Linear predictor method
    temporal_scheme = LF
    U_lf, x_lf, y_lf = Cauchy_problem_LP(F, t, Uo, temporal_scheme)

    # Plot 1: Phase space plot
    plt.plot(x_euler, y_euler, label='Euler')
    plt.plot(x_inv_euler, y_inv_euler, label='Inverse Euler')
    plt.plot(x_cn, y_cn, label='CN')
    plt.plot(x_rk4, y_rk4, label='RK4')
    plt.plot(x_lf, y_lf, label='LF')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.title('Harmonic Oscillator with different schemes')
    plt.show()

    # Plot 2: x vs t for each method
    plt.plot(t, x_euler, label='Euler')
    plt.plot(t, x_inv_euler, label='Inverse Euler')
    plt.plot(t, x_cn, label='CN')
    plt.plot(t, x_rk4, label='RK4')
    plt.plot(t, x_lf, label='LF')

    plt.xlabel('t')
    plt.ylabel('x')
    plt.ylim(-3, 3)
    plt.grid()
    plt.legend()
    plt.title('Harmonic Oscillator with different schemes: time domain')
    plt.show()


# Run the function
harmonic_oscillator_integration()



# Eigenvalues for the linear oscillator matrix A
A = array([[0, 1], [-1, 0]])
Eigenvalues = linalg.eigvals(A)
print("Eigenvalues A", Eigenvalues)

# Different time steps for analysis
dt_values = [1, 0.1, 0.01]

# Plotting stability regions of different methods
def plot_stability_regions(method, title):
    for dt in dt_values:
        x, y, rho = Stability_region(method, 100, -2.5, 0.5, -1.5, 1.5)
        plt.contour(x, y, rho.T, linspace(0, 1, 11))
        plt.scatter(real(Eigenvalues * dt), imag(Eigenvalues * dt), label=f'dt={dt}')

    plt.legend()
    plt.xlabel('Re(w)')
    plt.ylabel('Im(w)')
    plt.axis('equal')
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.title(title)
    plt.grid()
    plt.show()

# Stability regions for different methods
plot_stability_regions(Euler, 'Regions of absolute stability Euler')
plot_stability_regions(Inverse_Euler, 'Regions of absolute stability Inverse Euler')
plot_stability_regions(CN, 'Regions of absolute stability CN')
plot_stability_regions(RK4, 'Regions of absolute stability RK4')