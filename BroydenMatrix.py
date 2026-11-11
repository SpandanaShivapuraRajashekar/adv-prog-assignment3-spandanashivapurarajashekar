
import numpy as np
from scipy.linalg import svd, qr
from scipy.sparse.linalg import LinearOperator


class BroydenM(LinearOperator):

    __array_priority__ = 10000

    def __init__(self, d: int) -> None:
        super().__init__(dtype=np.float64, shape=(d, d))
        self._U = np.zeros((d, 0))
        self._Vt = np.zeros((0, d))

    def __len__(self):
        """Return the formal rank of the perturbation."""
        return self._Vt.shape[0]

    def append(self, u, v):
        """Append a rank-one update u @ v.T."""
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        v = np.asarray(v, dtype=np.float64).reshape(-1)

        if u.size != self.shape[0] or v.size != self.shape[1]:
            raise ValueError("Update vectors have incorrect dimensions.")

        self._U = np.column_stack((self._U, u))
        self._Vt = np.vstack((self._Vt, v))

    def todense(self):
        """Return the dense matrix representation."""
        return np.eye(self.shape[0]) + self._U @ self._Vt

    def _matvec(self, x):
        """Compute B @ x efficiently."""
        x = np.asarray(x).reshape(-1)
        return x + self._U @ (self._Vt @ x)

    def _rmatvec(self, x):
        """Compute B.T @ x efficiently."""
        x = np.asarray(x).reshape(-1)
        return x + self._Vt.T @ (self._U.T @ x)

    def __rmatmul__(self, x):
        """Support multiplication from the right: x @ A."""
        x = np.asarray(x)
        return x + (x @ self._U) @ self._Vt

    def prune(self, p=1, method='SVD'):
        """
        Reduce the formal rank of the Broyden update to p.

        Simple: Keep the newest p update pairs.
        SVD: Keep the best rank-p approximation in Frobenius norm.
        """
        if not isinstance(p, (int, np.integer)) or p < 0:
            raise ValueError("p must be a non-negative integer.")

        if p > len(self):
            raise ValueError("p cannot exceed the current rank.")

        if method == 'Simple':
            self._U = self._U[:, -p:] if p else self._U[:, :0]
            self._Vt = self._Vt[-p:, :] if p else self._Vt[:0, :]

        elif method == 'SVD':
            if p == 0:
                self._U = np.zeros((self.shape[0], 0))
                self._Vt = np.zeros((0, self.shape[1]))
                return

            if p == len(self):
                return

            # Reduced QR decompositions
            Qu, Ru = qr(self._U, mode='economic')
            Qv, Rv = qr(self._Vt.T, mode='economic')

            # SVD of the small core matrix
            P, s, Qt = svd(Ru @ Rv.T, full_matrices=False)

            # Retain the largest p singular values
            P = P[:, :p]
            Qt = Qt[:p, :]
            s = s[:p]

            sqrt_s = np.sqrt(s)

            # Balanced low-rank factors
            self._U = (Qu @ P) * sqrt_s
            self._Vt = sqrt_s[:, None] * (Qt @ Qv.T)

        else:
            raise ValueError("method must be 'Simple' or 'SVD'")

