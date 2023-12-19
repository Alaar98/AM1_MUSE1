from scipy.optimize import newton
from numpy import zeros, dot, float64
from numpy.linalg import norm

#Funcion del Esquema Temporal de Euler
def Euler( U, dt, t, F, q_value, tolerance_value): 

   return U + dt * F( U,t ) 

#Funcion del Esquema Temporal Runge-Kutta de orden 4
def RK4( U, dt, t, F, q_value, tolerance_value): 

     k1 = F( U, t)
     k2 = F( U + dt * k1/2, t + dt/2 )
     k3 = F( U + dt * k2/2, t + dt/2 )
     k4 = F( U + dt * k3,   t + dt   )
 
     return U + dt * ( k1 + 2*k2 + 2*k3 + k4 )/6

#Funcion del Esquema Temporal de Euler Inverso
def Inverse_Euler( U, dt, t, F, q_value, tolerance_value): 

   def  Residual_IE(X):  

          return  X - U - dt * F(X, dt + t) 

   return  newton( Residual_IE, U )
    
#Funcion del Esquema Temporal Implicito de Crank Nicolson
def Crank_Nicolson(U, dt, t, F, q_value, tolerance_value ): 

    def Residual_CN(X): 
         
         return  X - a - dt/2 *  F(X, t + dt)

    a = U  +  dt/2 * F( U, t)  
    return newton( Residual_CN, U )

#Funcion del Esquema Temporal Runge-Kutta Embebido
def Embedded_RK(U, t1, t2, F, q_value, tolerance_value):
    N_stages = {2: 2, 3: 4, 8: 13}

    try:
        Ns = N_stages[q_value]
    except KeyError:
        print(f"Error: El valor de q ({q_value}) no esta soportado.")
        return U  
    
    a = zeros((Ns, Ns), dtype=float64)
    b = zeros(Ns)
    bs = zeros(Ns)
    c = zeros(Ns)

    if Ns == 2:
        a[0, :] = [0, 0]
        a[1, :] = [1, 0]
        b[:] = [1 / 2, 1 / 2]
        bs[:] = [1, 0]
        c[:] = [0, 1]
    elif Ns == 13:
        c[:] = [ 0., 2./27, 1./9, 1./6, 5./12, 1./2, 5./6, 1./6, 2./3 , 1./3,   1., 0., 1.]

        a[0,:]  = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
        a[1,:]  = [ 2./27, 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0] 
        a[2,:]  = [ 1./36 , 1./12, 0., 0., 0., 0., 0.,  0.,0., 0., 0., 0., 0] 
        a[3,:]  = [ 1./24 , 0., 1./8 , 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
        a[4,:]  = [ 5./12, 0., -25./16, 25./16., 0., 0., 0., 0., 0., 0., 0., 0., 0]
        a[5,:]  = [ 1./20, 0., 0., 1./4, 1./5, 0., 0.,0., 0., 0., 0., 0., 0] 
        a[6,:]  = [-25./108, 0., 0., 125./108, -65./27, 125./54, 0., 0., 0., 0., 0., 0., 0] 
        a[7,:]  = [ 31./300, 0., 0., 0., 61./225, -2./9, 13./900, 0., 0., 0., 0., 0., 0] 
        a[8,:]  = [ 2., 0., 0., -53./6, 704./45, -107./9, 67./90, 3., 0., 0., 0., 0., 0] 
        a[9,:]  = [-91./108, 0., 0., 23./108, -976./135, 311./54, -19./60, 17./6, -1./12, 0., 0., 0., 0] 
        a[10,:] = [ 2383./4100, 0., 0., -341./164, 4496./1025, -301./82, 2133./4100, 45./82, 45./164, 18./41, 0., 0., 0] 
        a[11,:] = [ 3./205, 0., 0., 0., 0., -6./41, -3./205, -3./41, 3./41, 6./41, 0., 0., 0]
        a[12,:] = [ -1777./4100, 0., 0., -341./164, 4496./1025, -289./82, 2193./4100, 51./82, 33./164, 19./41, 0.,  1., 0]
      
        b[:]  = [ 41./840, 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 41./840, 0., 0.] 
        bs[:] = [ 0., 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 0., 41./840, 41./840]   

    k = RK_stages(F, U, t1, t2, a, c)
    Error = dot(b - bs, k)

    if norm(Error) == 0:
        # Evitar la division por cero
        dt_min = t2 - t1
    else:
        dt_min = min(t2 - t1, (t2 - t1) * (tolerance_value / norm(Error))**(1 / q_value))

    N = int((t2 - t1) / dt_min) + 1
    h = (t2 - t1) / N
    Uh = U.copy()

    for i in range(0, N):
        k = RK_stages(F, Uh, t1 + h * i, t1 + h * (i + 1), a, c)
        Uh += h * dot(b, k)

    return Uh

# Funcion que realiza los pasos de Runge-Kutta

def RK_stages(F, U, t1, t2, a, c):
    k = zeros((len(c), len(U)), dtype=float64)

    for i in range(len(c)):
        Up = U.copy()
        for j in range(len(c) - 1):
            Up =U+ (t2 - t1) * dot(a[i, :], k)

        k[i, :] = F(Up, t1 + c[i] * (t2 - t1))

    return k