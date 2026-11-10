# Advanced Programming
## Winter Semester 2026/27 -- Assignment 3

Note: Problems 1-5 will be autograded (with manual check for Problem 4, see below); Problem 6 will be graded manually.  For formulas and background information, please see the class notes.

1. Write a function `low_rank(A,k)` in file `svd_algorithms.py` which uses the SVD to compute the best approximation (in the Frobenius norm) to the matrix `A` with maximal rank `k`.

2. Write a function `pseudo_inverse(A,rtol=1e-15)` in file `svd_algorithms.py` which uses the SVD to compute the Moore–Penrose pseudo inverse of $A$, denoted $A^+$.  This matrix has the properties that if $x = A^+y$, then $y = Ax$.  Moreover, if $A$ has a nontrivial kernel, then $x$ is perpendicular to it.
   
    Note that you have to avoid inverting singular values that are smaller than `rtol` times the largest singular value, in order to avoid dividing by values that are indistinguishable from zero in floating point arithmetic.

3. Implement the class `BroydenM` in file `BroydenMatrix.py` as a subclass of `scipy.sparse.linalg.LinearOperator`.  This class should be an $O(nd)$ implementation of the Broyden matrix, where $n$ is the number of Broyden iterations (hence, the formal rank of the update matrix) and $d$ is the dimension of the problem.  Therefore, this class is more efficient than a full matrix representation when $n \ll d$.  The unit tests check for correct implementation of the methods `append`, `_matvec` and `_rmatvec` as well as `todense` which should yield a full matrix (as a two-dimensional array) that is equivalent to the `BroydenM` instance.  As the full matrix will use $O(d^2)$ memory, this method is only useful for testing and debugging.

4. Take your file `nd_roots.py` from Assignment 2, or, alternatively, the provided sample solution, and modify the implementation of the two Broyden methods such that it uses the class `BroydenM` to represent the approximate inverse Jacobian.

    This problem has to pass the autograder, but I will only award the points when the code has actually been changed to an $O(nd)$-implementation.  It is not sufficient to just use the old solution (which passes the unit tests!) but runs in $O(d^2)$-time.

5. In file `BroydenMatrix.py`, implement a method `prune(p=1, method='SVD')` to reduce the formal rank of the Broyden update to `p`.  When `method='Simple'`, do "simple pruning", i.e., simply remove all but the newest $p$ pairs of update vectors.  When `method='SVD'`, use SVD-based pruning as discussed in class.

6. In file `rootfinding_timing_integral.py`, compare the run-time of the Newton Method, the good Broyden method in full matrix representation, the good Broyden method using your `BroydenM` class, with and without pruning, for the integral equation test problem from Assignment 2.

Suggestion: make a slight adaptation of Assignment 2 Problem 6.
