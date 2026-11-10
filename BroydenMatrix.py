from numpy import newaxis, float64, eye, zeros, c_, r_
from scipy.linalg import svd, rq
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator

class BroydenM(LinearOperator):
   
    def __init__(self, d: int) -> None:
        super().__init__(dtype=float64, shape=(d, d))
        self._U = zeros((d,0))
        self._Vt = zeros((0,d))

    def __len__(self):
        """Return the (formal) rank of the perturbation"""
        return len(self._Vt)

    def append(self, u, v):
        # replace with your own code
        pass

    def todense(self):
        # replace with your own code
        pass

    def _matvec(self, x):
        # replace with your own code
        pass

    def _rmatvec(self, x):
        # replace with your own code

    def prune(self, p=1, method='SVD'):
        """
        If method='Simple', remove the oldest colums (rows) from
        U (Vt) until only p are left.  If method='SVD', use SVD-based
        pruning to rank p.
        """
        # replace with your own code
        pass

