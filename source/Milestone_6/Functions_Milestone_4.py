from numpy import abs, linspace, zeros, float64, max  

### Stability Regions ###


# Function to calculate stability regions for numerical methods
# Inputs:
#   scheme: Any of the numerical methods used to resolve the problem - from Temporal_Schemes.py
#   N: Number of points (normally 100)
#   x0, xf, y0, yf: Limit values of the plane
# Returns:
#   x, y, rho: x and y axes values, and stability values at each point in the plane


def calculate_stability_region(scheme, N, x0, xf, y0, yf): 
    x, y = linspace(x0, xf, N), linspace(y0, yf, N)
    rho = zeros((N, N), dtype=float64)
    
    for i in range(N):
        for j in range(N):
            w = complex(x[i], y[j])
            r = scheme(1., 0., 1., lambda u, t: w * u)
            rho[i, j] = abs(r)

    return x, y, rho 