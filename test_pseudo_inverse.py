from svd_algorithms import *
from numpy import array, allclose, vstack
from numpy.random import seed, rand
from numpy.linalg import inv

seed(1234)

A = rand(4,10)

x = rand(10)

def test_pseudo_inverse1():
    Apinv = pseudo_inverse(A)
    y = A@x
    xinv = Apinv@y
    assert allclose(A@xinv, y) and allclose(xinv, A.T@inv(A@A.T)@A@x)
    ###### xinv solves A@x=y   and xinv is normal to Ker(A)

B = A*1e20
C = vstack((B,B[0]+B[1]+B[2]+B[3]))
# C is a matrix with very large singular values and, in floating point
# arithmetic, a relatively small singular value which should be zero
# if it were computed in exact arithmetic
#
# The following test checks if the relative tolerance bound is handled
# correctly in the pseudo_inverse code.

def test_pseudo_inverse2():
    Cpinv = pseudo_inverse(C)
    y = C@x
    xinv = Cpinv@y
    assert allclose(C@xinv, y) # xinv solves C@x=y
    

def test_pseudo_inverse3():
    with open('svd_algorithms.py') as f:
        p = f.read()
        if p.find('pinv') >= 0:
            raise ValueError("You should not use the built-in pinv")
    
    
