
############################################################
###   Milestone 6: Lagrange points and their stability   ###
############################################################

from numpy import array, linspace, zeros, around, float64, real, imag, transpose
import numpy as np
import matplotlib.pyplot as plt
from random import random
from ODEs.Temporal_Schemes import Euler, CN, RK4, Inverse_Euler, LF
from ODEs.Temporal_Schemes import RKEmb
from Milestone_5.N_Body_function import N_Body
from Functions_Milestone_6 import CR3BP, Lpoints, Stability
from Milestone_4.Functions_Milestone_4 import Stability_region

# Main Inputs for Calculation
selected_LG = 5  # Lagrange Point to simulate in the final graphs (default: LP5)
orbit_calculation = 1  # Duration of orbit around a Lagrange Point (values between 1 and 10 for efficient computation)

# Cauchy Problem for Lagrange Points
def Cauchy_Lagrange(F, t, U0, temporal_scheme):
    Nv = len(U0)  # Number of columns needed
    N = len(t) - 1  # Number of rows needed
    U = np.zeros((N + 1, Nv), dtype=np.float64)
    U[0, :] = U0

    for i in range(N):
        U[i + 1, :] = temporal_scheme(U[i, :], t[i], t[i + 1] - t[i], F)

    return U

# Final Time and Number of Divisions
N = int(1e4)
t = np.linspace(0, orbit_calculation, N)
mu = 1.2151e-2  # Earth-Moon

# Function for the 3-body problem
def F(U, t):
    return CR3BP(U, mu)

# Lagrange Points based on close points
initial_LP = np.array([[0.1, 0, 0, 0], [1.01, 0, 0, 0], [-0.1, 0, 0, 0], [0.8, 0.6, 0, 0], [0.8, -0.6, 0, 0]])
Lagrange_Points = Lpoints(initial_LP, 5, mu)

# Display calculated Lagrange points with labels
for i, lag_point in enumerate(Lagrange_Points):
    label = f"L{i + 1}"  # Adjust numbering based on preference (e.g., L1, L2, etc.)
    print(f"{label}: {lag_point}")

# Generate initial conditions close to a Lagrange point
U0 = np.zeros(4)
U0[0:2] = Lagrange_Points[selected_LG - 1, :]
random_offset = 1e-4 * random()
U0 = U0 + random_offset

# Integration of the circular restricted 3-body problem using a temporal scheme
temporal_scheme = CN
U = Cauchy_Lagrange(F, t, U0, temporal_scheme)

# Evaluate stability in Lagrange points
for i in range(5):
    U0S = np.zeros(4)
    U0S[0:2] = Lagrange_Points[i, :]
    eigenvalues = Stability(U0S, mu)
    print(f"LP {i + 1} eigenvalue: {around(eigenvalues.real, 4)}")


#Lagrange points represent specific locations in space where objects placed there 
#tend to maintain their positions. These points occur where the gravitational forces 
#of two massive bodies balance the centripetal force required for a smaller object 
#to orbit with them. These positions are advantageous for spacecraft, as they enable 
#a reduction in the fuel needed to sustain their position.

#Among the five Lagrange points, three are deemed unstable (L1, L2, and L3). 
#They align along the imaginary line connecting the two larger celestial bodies. 
#The remaining two points, L4 and L5, are stable. These stable points form the vertices 
#of two equilateral triangles with the massive bodies. Notably, L4 precedes Earth's orbit, 
#while L5 follows it.