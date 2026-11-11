
from numpy.linalg import solve
from numpy import max, abs
from scipy.optimize import approx_fprime
from BroydenMatrix import BroydenM


def newton(f, x0, J=None, tol=1e-7, maxit=100):
    """
    Computes the root of function f using Newton's method.

    If a root is not found within maxit iterations, returns None.
    """
    x = x0.copy()
    f_old = f(x)

    if max(abs(f_old)) <= tol:
        return x

    for i in range(maxit):
        if J is None:
            J_current = approx_fprime(x, f)
        else:
            J_current = J(x)

        x -= solve(J_current, f_old)

        f_old = f(x)

        if max(abs(f_old)) <= tol:
            return x

    return None


def broyden_good(f, x0, tol=1e-7, maxit=100):
    """
    Computes the root using the Good Broyden method.

    The inverse Jacobian approximation starts as the identity
    matrix and is represented by BroydenM.

    Returns None if the method does not converge.
    """
    Jinv = BroydenM(len(x0))

    x_old = x0.copy()
    f_old = f(x_old)

    if max(abs(f_old)) < tol:
        return x_old

    for i in range(maxit):
        del_x = -Jinv @ f_old
        x_new = x_old + del_x

        f_new = f(x_new)

        if max(abs(f_new)) < tol:
            return x_new

        v = Jinv @ (f_new - f_old)

        denominator = del_x @ v

        if abs(denominator) < 1e-15:
            return None

        u = (del_x - v) / denominator
        w = Jinv.T @ del_x

        Jinv.append(u, w)

        x_old = x_new
        f_old = f_new

    return None


def broyden_bad(f, x0, tol=1e-7, maxit=100):
    """
    Computes the root using the Bad Broyden method.

    The inverse Jacobian approximation starts as the identity
    matrix and is represented by BroydenM.

    Returns None if the method does not converge.
    """
    Jinv = BroydenM(len(x0))

    x_old = x0.copy()
    f_old = f(x_old)

    if max(abs(f_old)) < tol:
        return x_old

    for i in range(maxit):
        del_x = -Jinv @ f_old
        x_new = x_old + del_x

        f_new = f(x_new)

        if max(abs(f_new)) < tol:
            return x_new

        del_f = f_new - f_old

        denominator = del_f @ del_f

        if denominator < 1e-15:
            return None

        u = (del_x - Jinv @ del_f) / denominator

        Jinv.append(u, del_f)

        x_old = x_new
        f_old = f_new

    return None
