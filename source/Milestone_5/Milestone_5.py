
#######################################
###   Milestone 5: N body problem   ###
#######################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Temporal_Schemes import  runge_kutta_4_integration
from N_Body_function import N_Body

# Function to solve the N-body problem using a Cauchy approach
def Cauchy_N_body(F, t, Uo, temporal_scheme):
    Nv = len(Uo)  # Number of columns needed
    N = len(t) - 1  # Number of rows needed
    U = np.zeros((N + 1, Nv), dtype=np.float64)
    U[0, :] = Uo

    for i in range(N):
        U[i + 1, :] = temporal_scheme(U[i, :], t[i], t[i + 1] - t[i], F)

    return U

# Initialize the initial positions and velocities of N bodies
def Initialize_N_body(Nc, Nb):
    Uo = np.zeros(2 * Nc * Nb)  # Column vector with rows as needed
    U1 = np.reshape(Uo, (Nb, Nc, 2))
    r0 = np.reshape(U1[:, :, 0], (Nb, Nc))
    v0 = np.reshape(U1[:, :, 1], (Nb, Nc))

    # Define initial positions and velocities for each body
    r0[0, :] = [2, 0, 0]
    v0[0, :] = [0, 0.8, 0]
    r0[1, :] = [-2, 0, 0]
    v0[1, :] = [0, -0.8, 0]
    r0[2, :] = [0, 2, 0]
    v0[2, :] = [-0.8, 0, 0]
    r0[3, :] = [0, -2, 0]
    v0[3, :] = [0.8, 0, 0]

    return Uo

# Define final time, number of divisions, and initial conditions
Nb = 4  # Number of bodies
Nc = 3  # Number of coordinates

N = 1000  # Time steps
tf = 16 * 3.14  # Final time
t = np.linspace(0, tf, N + 1)
Uo = Initialize_N_body(Nc, Nb)

# Function to define the force
def F(U, t):
    return N_Body(U, t, Nb, Nc)

temporal_scheme = runge_kutta_4_integration
U = Cauchy_N_body(F, t, Uo, temporal_scheme)

# Plotting 3D graph
r = np.reshape(U, (N + 1, Nb, Nc, 2))[:, :, :, 0]
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
colors = ["blue", "red", "green", "purple", "yellow", "orange", "black"]

for i in range(Nb):
    ax1.plot_wireframe(r[:, i, 0].reshape((-1, 1)), r[:, i, 1].reshape((-1, 1)), r[:, i, 2].reshape((-1, 1)),
                       color=colors[i])

plt.title("N-body problem")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
plt.grid()
plt.show()

# Plotting 2D graph
for i in range(Nb):
    plt.plot(r[:, i, 0], r[:, i, 1], color=colors[i])

plt.axis('equal')
plt.title("N-body problem X-Y Plane")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()