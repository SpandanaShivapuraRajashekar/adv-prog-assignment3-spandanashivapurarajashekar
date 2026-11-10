### This file is copied over from Assignment 2
###
### You need to modify this code to use the new class BroydenM instead
### of standard numpy arrays.
###
### The class BroydenM assumes that the initial inverse matrix is the
### identity matrix.  Therefore, you may remove the argument Jinv from
### the two functions implementing the Broyden methods.

from numpy.linalg import solve
from numpy import max, abs, eye, newaxis
from scipy.optimize import approx_fprime
from BroydenMatrix import BroydenM

from numpy.linalg import solve
from numpy import max, abs, eye, newaxis
from scipy.optimize import approx_fprime

def newton(f, x0, J=None, tol=1e-7, maxit=100):
    """
    Computes the root of function f that takes an array of length d
    and yields an array of length d, from starting array x0 via the
    Newton method. The Jacobian J must be supplied.  A root at x is
    found if max(abs((x))<tol.  Performs at most maxit iterations.  If
    a root is not found within maxit iterations, a RuntimeError is
    raised.
    """
    x = x0
    f_old = f(x)
    if max(abs(f_old)) <= tol:
        return x
    for i in range(maxit):
        if J is None:
            J = approx_fprime(x, f)
            x -= solve(J, f_old)
        else:
            x -= solve(J(x), f_old)
        f_old = f(x)
        if max(abs(f_old)) <= tol:
            print (i)
            return x

    raise RuntimeError(f"The Newton method failed to converge after {maxit} iterations")
    return None

def broyden_good(f, x0, Jinv = None, tol=1e-7, maxit=100):
    """
    Computes the root of function f that takes an array of length d
    and yields an array of length d, from starting array x0 via the
    Good Broyden method. The Jacobian J must be supplied.  A root at x is
    found if max(abs((x))<tol.  Performs at most maxit iterations.  If
    a root is not found within maxit iterations, a RuntimeError is
    raised.
    """
    if Jinv is None:
        Jinv = eye(len(x0))
    x_old = x0
    f_old = f(x0)
    if max(abs(f_old)) < tol:
        return x0

    for i in range(maxit):
        del_x = - Jinv @ f_old
        x_new = x_old + del_x
        f_new = f(x_new)
        if max(abs(f_new)) < tol:
            return x_new
        v = Jinv @ (f_new - f_old)
        Jinv += ((del_x - v)/(del_x @ v))[:,newaxis] * (del_x @ Jinv)[newaxis,:]
        x_old, f_old = x_new, f_new

    raise RuntimeError(f"The Good Broyden method failed to converge after {maxit} iterations")
    return None

def broyden_bad(f, x0, Jinv = None, tol=1e-7, maxit=100):
    """
    Computes the root of function f that takes an array of length d
    and yields an array of length d, from starting array x0 via the
    Bad Broyden method. The Jacobian J must be supplied.  A root at x is
    found if max(abs((x))<tol.  Performs at most maxit iterations.  If
    a root is not found within maxit iterations, a RuntimeError is
    raised.
    """
    if Jinv is None:
        Jinv = eye(len(x0))
    x_old = x0
    f_old = f(x0)
    if max(abs(f_old)) < tol:
        return x0

    for i in range(maxit):
        del_x = - Jinv @ f_old
        x_new = x_old + del_x
        f_new = f(x_new)
        if max(abs(f_new)) < tol:
            return x_new
        del_f = f_new - f_old
        Jinv += ((del_x - Jinv@del_f)/(del_f @ del_f))[:,newaxis] * del_f[newaxis,:]
        x_old, f_old = x_new, f_new

    raise RuntimeError(f"The Good Broyden method failed to converge after {maxit} iterations")
    return None
