from numpy import array, zeros, float64
import matplotlib.pyplot as plt

### CAUCHY ###
# Function to integrate a Cauchy problem using different numerical methods

# Inputs:
#   F(U,t): Function dU/dt = F(U,t) - from Physics.py
#   t: Time partition t (vector of length N+1)
#   dt: Time step
#   Uo: Initial condition at t=0
#   temporal_scheme: Any of the numerical methods used to resolve the problem - from Temporal_schemes.py
# Returns:
#   U: Matrix[Nv, N+1] - Nv state values at N+1 time steps
#   x: Values of the 1st row (x-axis)
#   y: Values of the 2nd row (y-axis)


def integrate_cauchy_problem(F, t, Uo, temporal_scheme):
    Nv = len(Uo)  # Number of rows needed
    N = len(t) - 1  # Number of columns needed
    U = zeros((Nv, N + 1), dtype=float64)
    U[:, 0] = Uo

    for i in range(N):
        U[:, i + 1] = temporal_scheme(U[:, i], t[i], t[i + 1] - t[i], F)

    x = U[0, :]  # Collect x values or values of the 1st row
    y = U[1, :]  # Collect y values or values of the 2nd row

    return U, x, y  


# Function to integrate a 3D Cauchy problem using different numerical methods
# Inputs and Returns similar to integrate_cauchy_problem, but extends to a 3D system
def integrate_cauchy_problem_3D(F, t, Uo, temporal_scheme):
    Nv = len(Uo)  # Number of rows needed
    N = len(t) - 1  # Number of columns needed
    U = zeros((Nv, N + 1), dtype=float64)
    U[:, 0] = Uo

    for i in range(N):
        U[:, i + 1] = temporal_scheme(U[:, i], t[i], t[i + 1] - t[i], F)

    x = U[0, :]  # Collect x values or values of the 1st row
    y = U[1, :]  # Collect y values or values of the 2nd row
    z = U[2, :]  # Collect z values or values of the 3rd row

    return U, x, y, z