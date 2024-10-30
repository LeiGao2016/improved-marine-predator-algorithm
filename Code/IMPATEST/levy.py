import math
import numpy as np

def levy(n, m, beta):
    # % used for Numerator
    num = ((math.gamma(1+beta)) * math.sin(math.pi*beta/2))
    # % used for Denominator
    den = (math.gamma((1+beta)/2))*beta * (2**((beta-1)/2))
    sigma_u = (num/den)**(1/beta)  # % Standard deviation
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    z = u/(abs(v)**(1/beta))

    return z