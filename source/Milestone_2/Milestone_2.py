
# Milestone 2: Prototypes to integrate orbits with functions

from numpy import array, linspace,zeros
from scipy import optimize
import matplotlib.pyplot as plt

def F(U):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x*2 + y*2)*1.5
    return array ( [ vx, vy, -x/mr, -y/mr] ) 

def Euler(U,t_1,t_2,F):
    return U+(t_2-t_1)*F(U,t_1)

def Crank_Nicholson(U,t_1,t_2,F):
    def Residual (X):
        return X-U-(t_2-t_1)*0.5*(F(U,t_1)+F(X,t_2))
    return optimize.newton (Residual,U)

def Cauchy_Problem (time_domain,Euler,F,U):
    for n in range (0,time_domain[len(time_domain)]):
        U[:,n+1] = Euler(U[:,n],time_domain[n],time_domain[n+1],F)
        return


N = 1000000
U = array( [1, 0, 0, 1] ) 
dt = 0.01
time_domain = linspace (0, N*dt, N+1)