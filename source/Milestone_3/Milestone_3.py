from numpy import array, zeros, float64, arange,linspace
from scipy.optimize import newton
import matplotlib.pyplot as plt

from Cauchy_Problem import integrate_cauchy_problem
from Temporal_Schemes import euler_integration, crank_nicolson_integration, runge_kutta_4_integration, inverse_euler_integration
from Orbits import Kepler
from Functions_Milestone_3 import calculate_richardson_error, calculate_convergence_ratio

# 1. Write a function to evaluate errors by means of Richardson: Refer to Richardson.py

# 2.Numerical error of different temporal schemes

### Richardson error##
# ------------------------------------------------------------------------------- #
#INPUTS:
    # F(U,t) : Function dU/dt = F(U,t) -> from Physics.py
    # t : time partition t (vector of length N)
    # Uo : initial condition at t=0
    # Temporal_scheme is any of the numerical methods used to resolve the problem -> from Temporal_schemes.py

#RETURN:
    # E: matrix[Nv, N+1], Nv state values at N time steps 
# ------------------------------------------------------------------------------- #

# Euler
F = Kepler
dt = 0.01
t = arange(0, 200, dt)  
Uo = array( [1,0,0,1] ) 
temporal_scheme = euler_integration

E_EULER = calculate_richardson_error(F, t, Uo, temporal_scheme)
plt.axis('equal')
plt.plot(t, E_EULER[0,:],label='EULER')


# CN
F = Kepler
dt = 0.01
t = arange(0, 200, dt)  
Uo = array( [1,0,0,1] ) 
temporal_scheme = crank_nicolson_integration

E_CN = calculate_richardson_error(F, t, Uo, temporal_scheme)
plt.plot(t, E_CN[0,:],label='CN')


# RK4
F = Kepler
dt = 0.01
t = arange(0, 200, dt)  
Uo = array( [1,0,0,1] ) 
temporal_scheme = runge_kutta_4_integration

E_RK4 = calculate_richardson_error(F, t, Uo, temporal_scheme)
plt.plot(t, E_RK4[0,:],label='RK4')


# Inverse Euler - It does not converge, so it is only evaluated in a shorter time domain
F = Kepler
dt = 0.1
t = arange(0, 2, dt)   
Uo = array( [1,0,0,1] ) 
temporal_scheme = inverse_euler_integration

E_IE = calculate_richardson_error(F, t, Uo, temporal_scheme)
plt.plot(t, E_IE[0,:],label='Inverse_Euler')

#Graph
plt.xlabel('t')
plt.ylabel('ERROR') 
plt.axis('equal')
plt.grid()
plt.legend()
plt.title('Numerical error of different schemes solving Kepler')
plt.show()


# 3. Write a function to evaluate the convergence rate of different temporal schemes: Refer to Richardson.py

# 4. Convergence rate of the different methods with the time step

### Convergence ratio ###
# ------------------------------------------------------------------------------- #
#INPUTS:
    # F(U,t) : Function dU/dt = F(U,t) -> from Physics.py
    # t : time partition t (vector of length N)
    # Uo : initial condition at t=0
    # Temporal_scheme is any of the numerical methods used to resolve the problem -> from Temporal_schemes.py
    # p: is the number of points in the graph 

#RETURN:
    # Elog and Nlog
# ------------------------------------------------------------------------------- #

# Euler
F = Kepler
N = 2000
t = linspace(0, 10, N) 
Uo = array( [1,0,0,1] )
temporal_scheme = euler_integration
p=5
#Graph
[log_E_EULER, log_N_EULER] = calculate_convergence_ratio(F, t, Uo, temporal_scheme,p)
print("EULER:",log_N_EULER, log_E_EULER)
plt.plot(log_N_EULER, log_E_EULER,label='EULER')
plt.xlabel('log_N')
plt.ylabel('log_E') 
plt.axis('equal')
plt.grid()
plt.legend()
plt.title('Convergence rate Euler  solving Kepler')
plt.show()


# CN
F = Kepler
N = 2000
t = linspace(0, 10, N) 
Uo = array( [1,0,0,1] )
temporal_scheme = crank_nicolson_integration
p=5
#Graph
[log_E_CN, log_N_CN] = calculate_convergence_ratio(F, t, Uo, temporal_scheme,p)
print("CN:",log_N_CN, log_E_CN)
plt.plot(log_N_CN, log_E_CN,label='CN')
plt.xlabel('log_N')
plt.ylabel('log_E') 
plt.axis('equal')
plt.grid()
plt.legend()
plt.title('Convergence rate CN method solving Kepler')
plt.show()


# RK4
F = Kepler
N = 2000
t = linspace(0, 10, N) 
Uo = array( [1,0,0,1] )
temporal_scheme = runge_kutta_4_integration
p=5
#Graph
[log_E_RK4, log_N_RK4] = calculate_convergence_ratio(F, t, Uo, temporal_scheme,p)
print("RK4:",log_N_RK4, log_E_RK4)
plt.plot(log_N_RK4, log_E_RK4,label='RK4')
plt.xlabel('log_N')
plt.ylabel('log_E') 
plt.axis('equal')
plt.grid()
plt.legend()
plt.title('Convergence rate RK4 method solving Kepler')
plt.show()