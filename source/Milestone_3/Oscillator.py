from numpy import array
def Oscillator(U, t): # Initial value of U: U = array( [1,0] ) 

    x = U[0]; dxdt = U[1]; 

    return  array( [ dxdt, -x ] ) 