from numpy import linspace, size, zeros, log10, float64, ones, vstack, array
from numpy.linalg import norm, lstsq
from ODEs.Temporal_Schemes import Euler, CN, RK4, Inverse_Euler

# Solve Cauchy problem in the context of the Richardson error method
def solve_cauchy_problem_richardson_error(F, t, Uo, temporal_scheme):
    Nv = len(Uo)
    N = len(t) - 1
    U = zeros((Nv, N + 1), dtype=float64)
    U[:, 0] = Uo

    for i in range(N):
        U[:, i + 1] = temporal_scheme(U[:, i], t[i], t[i + 1] - t[i], F)

    x_values = U[0, :]
    y_values = U[1, :]

    return U

# Calculate Richardson error using two-step solutions
def calculate_richardson_error(F, t, Uo, temporal_scheme):
    Nv = len(Uo)
    N = len(t) - 1
    t1 = t
    t2 = zeros(2 * N + 1)
    E = zeros([Nv, N + 1], dtype=float64)

    for i in range(N + 1):
        t2[2 * i] = t1[i]

    for i in range(N):
        t2[2 * i + 1] = (t1[i] + t1[i + 1]) / 2

    U1 = solve_cauchy_problem_richardson_error(F, t1, Uo, temporal_scheme)
    U2 = solve_cauchy_problem_richardson_error(F, t2, Uo, temporal_scheme)

    if temporal_scheme == RK4:
        q = 4
    elif temporal_scheme == CN:
        q = 2
    else:
        q = 1

    for i in range(N + 1):
        E[:, i] = (U2[:, 2 * i] - U1[:, i]) / (1 - 1. / (2 ** q))

    return E

# Calculate convergence ratio
def calculate_convergence_ratio(F, t, Uo, temporal_scheme, p):
    Elog = zeros(p)
    Nlog = zeros(p)
    N = len(t) - 1
    t1 = t
    U1 = solve_cauchy_problem_richardson_error(F, t1, Uo, temporal_scheme)

    if temporal_scheme == RK4:
        q = 4
    elif temporal_scheme == CN:
        q = 2
    else:
        q = 1

    for i in range(p):
        N = 2 * N
        t2 = array(zeros(N + 1))
        t2[0:N + 1:2] = t1
        t2[1:N:2] = (t1[1:int(N / 2) + 1] + t1[0:int(N / 2)]) / 2
        U2 = solve_cauchy_problem_richardson_error(F, t2, Uo, temporal_scheme)

        E = norm(U2[:, N] - U1[:, int(N / 2)])
        Elog[i] = log10(E) - log10(1 - 1 / 2 ** q)
        Nlog[i] = log10(N)
        t1 = t2
        U1 = U2

    return [Elog, Nlog]