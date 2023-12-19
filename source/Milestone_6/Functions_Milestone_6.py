from numpy.linalg import norm, eig
from numpy import array, zeros, sqrt, size
from scipy.optimize import fsolve

# Function defining the equations of motion in the Circular Restricted Three-Body Problem (CR3BP)
def CR3BP(U, mu):
    r = U[0:2]  # Position vector
    drdt = U[2:4]  # Velocity vector

    p1 = sqrt((r[0] + mu) ** 2 + r[1] ** 2)
    p2 = sqrt((r[0] - 1 + mu) ** 2 + r[1] ** 2)

    dvdt_1 = -(1 - mu) * (r[0] + mu) / (p1 ** 3) - mu * (r[0] - 1 + mu) / (p2 ** 3)
    dvdt_2 = -(1 - mu) * r[1] / (p1 ** 3) - mu * r[1] / (p2 ** 3)

    return array([drdt[0], drdt[1], 2 * drdt[1] + r[0] + dvdt_1, -2 * drdt[0] + r[1] + dvdt_2])

# Function to find the Lagrange Points in the Circular Restricted Three-Body Problem (CR3BP)
def lagrange_points(U_0, Np, mu):
    LP = zeros([5, 2])

    def F(Y):
        X = zeros(4)
        X[0:2] = Y
        X[2:4] = 0
        FX = CR3BP(X, mu)
        return FX[2:4]

    for i in range(Np):
        LP[i, :] = fsolve(F, U_0[i, 0:2])

    return LP

# Function to calculate stability eigenvalues and eigenvectors in the Circular Restricted Three-Body Problem (CR3BP)
def stability(U_0, mu):

    def F(Y):
        return CR3BP(Y, mu)

    A = jacobian_matrix(F, U_0)
    values, vectors = eig(A)

    return values

# Function to calculate the Jacobian matrix
def jacobian_matrix(F, U):
    N = size(U)
    Jac = zeros([N, N])
    t = 1e-10

    for i in range(N):
        xj = zeros(N)
        xj[i] = t
        Jac[:, i] = (F(U + xj) - F(U - xj)) / (2 * t)
    return Jac