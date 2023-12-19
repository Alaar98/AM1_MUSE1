# Milestone 1: Prototypes to integrate orbits without functions

import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

# Initial conditions
initial_state = np.array([1, 0, 0, 1])  # Initial state vector [x, y, dx/dt, dy/dt]

# Function defining the differential equation for the Keplerian orbit
def kepler_equation(state, time):
    x, y, dxdt, dydt = state
    distance = (x**2 + y**2)**1.5  # Denominator in the equations
    return np.array([dxdt, dydt, -x / distance, -y / distance])

# Euler Temporal Scheme
def euler_method(U, delta_t, time, F):
    return U + delta_t * F(U, time)

# Runge-Kutta Temporal Scheme (4th order)
def runge_kutta_4th_order(state, delta_t, time, equation):
    k1 = equation(state, time)
    k2 = equation(state + delta_t * k1 / 2, time + delta_t / 2)
    k3 = equation(state + delta_t * k2 / 2, time + delta_t / 2)
    k4 = equation(state + delta_t * k3, time + delta_t)
    return state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Crank-Nicolson Temporal Scheme
def crank_nicolson(U, delta_t, time, F):
    def residual_cn(X):
        return X - a - delta_t / 2 * F(X, time + delta_t)

    a = U + delta_t / 2 * F(U, time)
    return newton(residual_cn, U)

# Function to simulate orbit using the specified method and parameters
def simulate_orbit(method, num_steps, time_step, initial_state, title):
    positions_x = np.zeros(num_steps)
    positions_y = np.zeros(num_steps)
    time_array = np.zeros(num_steps)

    positions_x[0] = initial_state[0]
    positions_y[0] = initial_state[1]
    time_array[0] = 0

    # Iterating over time steps to compute positions
    for i in range(1, num_steps):
        time_array[i] = time_step * i
        initial_state = method(initial_state, time_step, time_array[i], kepler_equation)
        positions_x[i] = initial_state[0]
        positions_y[i] = initial_state[1]

    # Plotting the orbit
    plt.plot(positions_x, positions_y)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(title)
    plt.show()

# Simulations using different methods

# Euler method simulation
num_steps = 1000
time_step = 0.01
simulate_orbit(euler_method, num_steps, time_step, initial_state, 'Kepler Orbit (EULER METHOD) for Steps={} and Time Step={}'.format(num_steps, time_step))

# Runge-Kutta 4th order method simulation
num_steps = 100
time_step = 0.1
simulate_orbit(runge_kutta_4th_order, num_steps, time_step, initial_state, 'Kepler Orbit (RK4 METHOD) for Steps={} and Time Step={}'.format(num_steps, time_step))

# Crank-Nicolson method simulation
num_steps = 100
time_step = 0.1
simulate_orbit(crank_nicolson, num_steps, time_step, initial_state, 'Kepler Orbit (C-N METHOD) for Steps={} and Time Step={}'.format(num_steps, time_step))