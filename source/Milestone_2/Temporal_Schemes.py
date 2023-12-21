from numpy import array, zeros, matmul
from numpy.linalg import norm
from scipy.optimize import newton
import matplotlib.pyplot as plt


# Function to perform one step of Euler integration
def euler_integration(U, t, dt, F): 
    return U + dt * F(U, t)


# Function to perform one step of Crank-Nicolson integration
def crank_nicolson_integration(U, t, dt, F): 
    def residual_CN(X): 
        return X - U - dt/2 * F(U, t) - dt/2 * F(X, t + dt)
    return newton(residual_CN, U)


# Function to perform one step of Runge-Kutta 4th order integration
def runge_kutta_4_integration(U, t, dt, F): 
    k1 = F(U, t)
    k2 = F(U + dt * k1 / 2, t + dt / 2)
    k3 = F(U + dt * k2 / 2, t + dt / 2)
    k4 = F(U + dt * k3, t + dt)
    return U + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Function to perform one step of Inverse Euler integration
def inverse_euler_integration(U, t, dt, F): 
    def residual_Eu(X): 
        return X - U - dt * F(X, t + dt)
    return newton(residual_Eu, U)


# Function to perform one step of Leapfrog integration
def leapfrog_integration(U2, U1, t, dt, F):
    return U1 + 2 * dt * F(U2, t)


# Function for adaptive Runge-Kutta embedded method
def adaptive_RK_emb(U, t, dt, F):
    # Set tolerance for error estimation
    tol = 1e-9
    # Obtain coefficients and orders for the Butcher array
    orders, Ns, a, b, bs, c = ButcherArray()
    # Estimate state at two different orders
    est1 = perform_RK(1, U, t, dt, F) 
    est2 = perform_RK(2, U, t, dt, F) 
    # Calculate optimal step size
    h = min(dt, calculate_step_size(est1 - est2, tol, dt, min(orders)))
    N_n = int(dt / h) + 1
    n_dt = dt / N_n
    est1 = U
    est2 = U

    # Perform multiple steps with the adaptive step size
    for i in range(N_n):
        time = t + i * dt / int(N_n)
        est1 = est2
        est2 = perform_RK(1, est1, time, n_dt, F)

    final_state = est2
    ierr = 0

    return final_state

# Function to perform one step of Runge-Kutta integration
def perform_RK(order, U1, t, dt, F):
    # Obtain coefficients and orders for the Butcher array
    orders, Ns, a, b, bs, c = ButcherArray()
    k = zeros([Ns, len(U1)])
    k[0, :] = F(U1, t + c[0] * dt)

    if order == 1: 
        for i in range(1, Ns):
            Up = U1
            for j in range(i):
                Up = Up + dt * a[i, j] * k[j, :]
            k[i, :] = F(Up, t + c[i] * dt)
        U2 = U1 + dt * matmul(b, k)

    elif order == 2:
        for i in range(1, Ns):
            Up = U1
            for j in range(i):
                Up = Up + dt * a[i, j] * k[j, :]
            k[i, :] = F(Up, t + c[i] * dt)
        U2 = U1 + dt * matmul(bs, k)

    return U2

# Function to calculate the optimal step size based on the estimated error
def calculate_step_size(dU, tol, dt, orders): 
    error = norm(dU)

    if error > tol:
        step_size = dt * (tol / error) ** (1 / (orders + 1))
    else:
        step_size = dt

    return step_size


# Function to define the Butcher array coefficients for a specific Runge-Kutta method
def ButcherArray(): 
    orders = [2, 1]
    Ns = 2 

    a = zeros([Ns, Ns - 1])
    b = zeros([Ns])
    bs = zeros([Ns])
    c = zeros([Ns])

    c = [0., 1.]
    a[0, :] = [0.]
    a[1, :] = [1.]
    b[:] = [1./2, 1./2]
    bs[:] = [1., 0.]

    return orders, Ns, a, b, bs, c