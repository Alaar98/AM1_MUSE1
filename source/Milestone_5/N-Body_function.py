from numpy import reshape, zeros
from numpy.linalg import norm

#-----------------------------------------------------------------
#  dvi/dt = - G m sum_j (ri- rj) / | ri -rj |**3, dridt = vi 
#----------------------------------------------------------------- 

def N_Body(U, t, Nb, Nc): 
    F =  zeros(len(U))
    Us  = reshape(U,(Nb, Nc, 2))
    dUs = reshape(F,(Nb, Nc, 2))

    r = reshape(Us[:,:,0],(Nb, Nc))
    v = reshape(Us[:,:,1],(Nb, Nc))
    
    drdt = reshape(dUs[:,:,0], (Nb,Nc))
    dvdt = reshape(dUs[:,:,1], (Nb,Nc))
    
    dvdt[:,:] = 0
    
    for i in range(Nb):
        drdt[i,:] = v[i,:]
        for j in range(Nb):
            if j != i:
                dvdt[i,:] = dvdt[i,:] + (r[j,:]-r[i,:])/norm(r[j,:]-r[i,:])**3 
    
    return F