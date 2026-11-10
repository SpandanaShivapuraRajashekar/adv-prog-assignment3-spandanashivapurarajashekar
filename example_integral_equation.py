### Copied over from Assignment 2
###
### No need to change this code.

from numpy import eye, newaxis, arange

def setup_integral_equation(d, a=0.5):
    
    h = 1/d
    t = h*arange(1,d+1)

    M = arange(d)[:,newaxis] >= arange(d)[newaxis,:]
    H = ((1-t)[:,newaxis]*t[newaxis,:])*M + \
        (t[:,newaxis]*(1-t)[newaxis,:])*(1-M)
    H *= a*h

    def f(x):
        return x + H@(x+t+1)**3

    def J(x):
        return eye(d) + H * (3*(x+t+1)**2)[newaxis,:]

    return f, J

