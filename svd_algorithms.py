
from numpy.linalg import svd
from numpy import zeros, diag


def low_rank(A, k):
    """
    Computes the best rank-k approximation of A using SVD.
    """
    U, s, Vt = svd(A, full_matrices=False)

    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]

    B = Uk @ diag(sk) @ Vtk

    return B


def pseudo_inverse(A, rtol=1e-15):
    """
    Computes the Moore-Penrose pseudoinverse using SVD.
    """
    U, s, Vt = svd(A, full_matrices=False)

    s_inv = zeros(len(s))

    if len(s) > 0:
        cutoff = rtol * s[0]

        for i in range(len(s)):
            if s[i] > cutoff:
                s_inv[i] = 1.0 / s[i]

    A_inverse = Vt.T @ diag(s_inv) @ U.T

    return A_inverse
