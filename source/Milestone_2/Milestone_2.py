
######################################################################
###   Milestone 2: Prototypes to integrate orbits with functions   ###
######################################################################

from numpy import array, zeros
import matplotlib.pyplot as plt
from ODEs.Temporal_Schemes import Euler, CN, RK4, Inverse_Euler
from ODEs.Orbits import Kepler

# 1. Function to integrate one step using Euler method

# 2. Function to integrate one step using Crank-Nicolson method

# 3. Function to integrate one step using RK4 method

# 4. Function to integrate one step using Inverse Euler method

# 5. Function to integrate a Cauchy problem

# 6. Function to express the force of the Kepler movement

# 7. Integrate a Kepler with different schemes and explain the results

def plot_kepler_orbit(N, dt, method):
    U = array([1, 0, 0, 1])  # Initial value of U
    x = zeros(N)
    y = zeros(N)
    t = zeros(N)
    
    x[0] = U[0]
    y[0] = U[1]
    t[0] = 0

    for i in range(1, N):
        t[i] = dt * i
        if method == 'Euler':
            U = Euler(U, t, dt, Kepler)
        elif method == 'CN':
            U = CN(U, t, dt, Kepler)
        elif method == 'RK4':
            U = RK4(U, t, dt, Kepler)
        elif method == 'Inverse_Euler':
            U = Inverse_Euler(U, t, dt, Kepler)
            
        x[i] = U[0]
        y[i] = U[1]
    
    plt.plot(x, y)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Kepler Orbit ({method} METHOD) for N={N} and dt={dt}')
    plt.show()

# Kepler with Euler method & print
plot_kepler_orbit(10000, 0.01, 'Euler')

# Kepler with CN method & print
plot_kepler_orbit(100, 0.1, 'CN')

# Kepler with RK4 method & print
plot_kepler_orbit(100, 0.1, 'RK4')

# Kepler with Inverse Euler method & print
plot_kepler_orbit(20, 0.1, 'Inverse_Euler')

# 8. Increase and decrease the time step and explain the results (Refer to Milestone 1)
