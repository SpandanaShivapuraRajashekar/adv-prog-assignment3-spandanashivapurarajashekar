import numpy as np
from nd_roots import *

### From Assignment 2 test_newton.py

def f(x):
    return np.r_[x[0] + x[1] + x[2] + 2,
                 x[1]*x[2] + x[0]*x[2] + x[0]*x[1] + 7,
                 x[0]*x[1]*x[2] + 1]

def J(x):
    return np.array([[1, 1, 1],
                     [x[1]+x[2], x[0]+x[2], x[0]+x[1]],
                     [x[1]*x[2], x[0]*x[2], x[0]*x[1]]])

def test_newton():
    r = np.roots([1, 2, -7, 1])
    x = newton(f, [1, 2, 3], J=J)
    r.sort()
    x.sort()
    assert np.allclose(r,x)
    
def g(x):
    return np.r_[x[0]**2 + x[1]**2 + 1,
                 x[0] + x[1] - 1]

def K(x):
    return np.array([[2*x[0], 2*x[1]],
                     [1, 1]])

def test_newton_no_root():
    assert newton(g, [3, 2], J=K) is None

### From Assignment 2 test_broyden_good.py

def test_broyden_good():
    r = np.roots([1, 2, -7, 1])
    x = broyden_good(f, [-3.9, 1.7, 0.15])
    r.sort()
    x.sort()
    assert np.allclose(r,x)
    
def test_broyden_good_no_root():
    assert broyden_good(g, [3, 2]) is None

### From Assignment 2 test_broyden_bad.py

def test_broyden_bad():
    r = np.roots([1, 2, -7, 1])
    x = broyden_bad(f, [-3.9, 1.7, 0.15])
    r.sort()
    x.sort()
    assert np.allclose(r,x)
    
def test_broyden_bad_no_root():
    assert broyden_bad(g, [3, 2]) is None
