from numpy import array, zeros, linspace
import numpy as np
import matplotlib.pyplot as plt
from Cauchy_problem import Cauchy_problem, Cauchy_Problem_mod
from Temporal_schemes import Euler, Inverse_Euler, Crank_Nicolson, RK4, Embedded_RK
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module
from time import process_time
import random

# ODE definition
def Arenstorf(U,t):
    x = U[0]; y = U[1]; Vx = U[2]; Vy = U[3]
    mu_moon = 0.012277471 #Gravitational parameter of the moon
    mu_earth = 1-mu_moon #Gravitational parameter of the Earth
    D1 = np.power(((x+mu_moon)**2 + y**2),1.5)
    D2 = np.power(((x-mu_earth)**2 + y**2),1.5)
    return array([Vx, Vy, x+(2*Vy)-(mu_earth*((x+mu_moon)/D1))-(mu_moon*((x-mu_earth)/D2)), y-(2*Vx)-(mu_earth*(y/D1))-(mu_moon*(y/D2))])             

#----------------------------------------------------------
#Initial Conditions
U0 = array([0.994, 0, 0, -2.00158510637908252240537862224])

#Parameters definition
T = 17.0652165601579625588917206249
N = 100000
dt = T/N
t = linspace(0,T,N)
x = array(zeros(N))
y = array(zeros(N))
x[0] = U0[0]
y[0] = U0[1]
q_value = 8  # Elige el valor de q segun tus necesidades
tolerance_value = 1e-2  # Elige la tolerancia segun tus necesidades

scheme = array([Crank_Nicolson,Euler,RK4,Embedded_RK])
start = array(zeros(len(scheme)))
finish = array(zeros(len(scheme)))
CPU_time = array(zeros(len(scheme)))
for i in range(len(scheme)):
    start[i] = process_time()
    if scheme[i] == Embedded_RK:
        sol = Cauchy_Problem_mod (t, scheme[i] , Arenstorf, U0, q_value, tolerance_value)
    else:
        sol = Cauchy_problem (Arenstorf, t, U0, scheme[i], q_value, tolerance_value)
    finish[i] = process_time()
    CPU_time [i] = finish[i]-start[i]
    print("Cauchy_Problem, CPU Time=",CPU_time [i]," seconds.") 


#--------------------------------------------------
    
plt.figure(figsize=(8, 6))  # Tamaño del gráfico (opcional)
colores = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(scheme))]
scheme_labels = ['Crank_Nicolson','Euler','RK4','RK Embebido']

plt.bar(scheme_labels, CPU_time, color=colores)  # Crear barras con los datos

plt.xlabel('Esquema Temporal')  # Etiqueta para el eje x
plt.ylabel('CPU_Time [s]')  # Etiqueta para el eje y
plt.title('Gráfico de Barras del Tiempo de Computación')  # Título del gráfico

plt.grid(True)  # Agregar una cuadrícula (opcional)
plt.tight_layout()  # Ajustar diseño

plt.show()  # Mostrar el gráfico