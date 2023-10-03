# Hito 1: Integracion de orbitas sin funciones

from numpy import array, linspace,zeros
from scipy import optimize
import matplotlib.pyplot as plt


# Integracion orbitas de Kepler con el metodo de Euler

# Las orbitas de Euler no tienen solucion, su error se puede paliar pero no eliminar
def F_Kepler(U):
    x, y, vx, vy = U[0], U[1], U[2], U[3]
    mr = (x*2 + y*2)*1.5
    return array ( [ vx, vy, -x/mr, -y/mr] ) 

# Mas vueltas
N = 1000000
U = array( [1, 0, 0, 1] ) 
dt = 0.01
t = linspace (0, N*dt, N+1)

x = array(zeros(N))
y = array(zeros(N))
x[0] = U[0]
y[0] = U[1]

for i in range (0, N):
    F = F_Kepler (U)
    U = U + dt * F
    x [i] = U [0]
    y [i] = U [1]
    
# Vamos a crear un array para almacenar la integracion y luego dibujarla
plt.axis('equal')
plt.plot(x,y)
plt.title ('Metodo de Euler')
plt.xlabel('Eje x')
plt.ylabel('Eje y')
plt.show()

# Integracion orbitas de Kepler con el metodo de Crank-Nicolson
    
def Crank_Nicolson(U, dt, N, F): 

    def Residual_CN(X): 
         
         return  X - a - dt/2 *  F_Kepler(X)
     
    a = U  +  dt/2 * F_Kepler(U)  
    return optimize.newton( Residual_CN, U )

xcn = array(zeros(N))
ycn = array(zeros(N))
xcn[0] = U[0]
ycn[0] = U[1]

for j in range(0,N):
    U = Crank_Nicolson (U,dt,N,F)
    xcn[j] = U[0]
    ycn[j] = U[1]

plt.axis('equal')    
plt.plot(xcn,ycn)
plt.title ('Metodo de Crank-Nicolson')
plt.xlabel('Eje x')
plt.ylabel('Eje y')
plt.show()
# Integracion orbitas de Kepler con el metodo Runge-Kutta de orden 4

def RK4 (U,dt):
    
    k1 = dt * F_Kepler (U)
    k2 = dt * F_Kepler (U + 0.5*k1)
    k3 = dt * F_Kepler (U + 0.5*k2)
    k4 = dt * F_Kepler (U + k3)
    
    return U + (k1 + 2*k2 + 2*k3 + k4)/6

xr4 = array(zeros(N))
yr4 = array(zeros(N))
xr4[0] = U[0]
yr4[0] = U[1]

for k in range(0,N):
    U = RK4 (U,dt)
    xr4[k] = U[0]
    yr4[k] = U[1]
    
plt.axis('equal')    
plt.plot(xr4,yr4)
plt.title ('Metodo de Runge-Kutta orden 4')
plt.xlabel('Eje x')
plt.ylabel('Eje y')
plt.show()

# Con RK4 tenemos el mismo problema que con Euler, en cambio, el error tarda mas N es aparecer
