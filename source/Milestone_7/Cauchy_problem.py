from  numpy import array, zeros, reshape, float64

#Problema de Cauchy 
def Cauchy_problem( F, t, U0, Temporal_scheme, q_value, tolerance_value): 



 N, Nv=  len(t)-1, len(U0)
 U = zeros( (N+1, Nv), dtype=float64) 

 U[0,:] = U0
 for n in range(N):

     U[n+1,:] = Temporal_scheme( U[n, :], t[n+1] - t[n], t[n],  F, q_value, tolerance_value) 

 return U

#Problema de Cauchy para el Runge-Kutta embebido
def Cauchy_Problem_mod (time_domain, temporal_scheme, F, U0, q_value, tolerance_value):
    N_t = len(time_domain) - 1
    N_ve = len(U0)
    U = zeros((N_t + 1, N_ve), dtype=float64)
    U[0, :] = U0

    for n in range(0, N_t):
        U[n + 1, :] = temporal_scheme(U[n, :], time_domain[n], time_domain[n + 1], F, q_value, tolerance_value)

    return U