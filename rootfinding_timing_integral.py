
import numpy as np
import matplotlib.pyplot as plt
import time

from example_integral_equation import setup_integral_equation
from nd_roots import newton
from BroydenMatrix import BroydenM


# Parameters
N = 5
a = 2
tol = 1e-13
i_max = 8
maxit = 100

# Timing arrays
t_newton = np.zeros(i_max)
t_broyden_good_full = np.zeros(i_max)
t_broyden_good_linop = np.zeros(i_max)
t_broyden_good_simple = np.zeros(i_max)
t_broyden_good_svd = np.zeros(i_max)

# Dimensions
dd = 20 * 2**np.arange(i_max)


def good_broyden(f, x0, representation, pruning=None):
    """Good Broyden with dense or low-rank inverse Jacobian."""

    d = len(x0)

    if representation == "full":
        Jinv = np.eye(d)
    else:
        Jinv = BroydenM(d)

    x = x0.copy()
    fx = f(x)

    if np.max(np.abs(fx)) < tol:
        return x

    for iteration in range(maxit):

        # Calculate the next step
        dx = -(Jinv @ fx)
        x_new = x + dx
        fx_new = f(x_new)

        # Check convergence
        if np.max(np.abs(fx_new)) < tol:
            return x_new

        # Good Broyden update
        v = Jinv @ (fx_new - fx)
        denominator = dx @ v

        # Scale-aware denominator check
        denominator_scale = (
            np.linalg.norm(dx) * np.linalg.norm(v)
        )

        if abs(denominator) <= (
            np.finfo(float).eps * denominator_scale
        ):
            print(
                f"Numerically zero denominator: d={d}, "
                f"iteration={iteration}, "
                f"denominator={denominator:.6e}, "
                f"residual={np.max(np.abs(fx_new)):.6e}"
            )
            return None

        u = (dx - v) / denominator
        w = Jinv.T @ dx

        # Update inverse Jacobian
        if representation == "full":
            Jinv += np.outer(u, w)

        else:
            Jinv.append(u, w)

            # Apply pruning if required
            if pruning is not None and len(Jinv) > N:
                print(
                    f"{pruning} pruning triggered: "
                    f"d={d}, rank={len(Jinv)}"
                )

                Jinv.prune(N, method=pruning)

        x = x_new
        fx = fx_new

    print(
        f"Not converged: d={d}, "
        f"final residual={np.max(np.abs(fx)):.6e}, "
        f"iterations={maxit}"
    )

    return None


def measure_time(method, f, x0):
    """Measure execution time and return solution."""

    start = time.process_time()
    solution = method(f, x0)
    elapsed = time.process_time() - start

    return elapsed, solution


# Run all methods
for i, d in enumerate(dd):

    print("\nDimension d =", d)

    f, J = setup_integral_equation(d, a=a)
    x0 = np.zeros(d)

    # 1. Newton
    t_newton[i], sol_newton = measure_time(
        lambda f, x: newton(
            f, x, J=J, tol=tol, maxit=maxit
        ),
        f, x0
    )

    # 2. Good Broyden: full matrix
    t_broyden_good_full[i], sol_full = measure_time(
        lambda f, x: good_broyden(
            f, x, "full"
        ),
        f, x0
    )

    # 3. Good Broyden: LinearOperator
    t_broyden_good_linop[i], sol_linop = measure_time(
        lambda f, x: good_broyden(
            f, x, "linop"
        ),
        f, x0
    )

    # 4. Good Broyden: Simple pruning
    t_broyden_good_simple[i], sol_simple = measure_time(
        lambda f, x: good_broyden(
            f, x, "linop", pruning="Simple"
        ),
        f, x0
    )

    # 5. Good Broyden: SVD pruning
    t_broyden_good_svd[i], sol_svd = measure_time(
        lambda f, x: good_broyden(
            f, x, "linop", pruning="SVD"
        ),
        f, x0
    )

    # Report convergence
    results = {
        "Newton": sol_newton,
        "Good Broyden (full)": sol_full,
        "Good Broyden (LinearOperator)": sol_linop,
        "Good Broyden (Simple pruning)": sol_simple,
        "Good Broyden (SVD pruning)": sol_svd
    }

    for name, solution in results.items():

        if solution is not None:
            residual = np.max(np.abs(f(solution)))
            status = f"Converged (residual={residual:.3e})"
        else:
            status = "Did not converge"

        print(f"  {name}: {status}")


# Plot timing results
plt.figure(figsize=(9, 6))

plt.loglog(
    dd, t_newton, "o-",
    label="Newton"
)

plt.loglog(
    dd, t_broyden_good_full, "o-",
    label="Good Broyden (full)"
)

plt.loglog(
    dd, t_broyden_good_linop, "o-",
    label="Good Broyden (LinearOperator)"
)

plt.loglog(
    dd, t_broyden_good_simple, "o-",
    label="Good Broyden (Simple pruning)"
)

plt.loglog(
    dd, t_broyden_good_svd, "o-",
    label="Good Broyden (SVD pruning)"
)

plt.title(f"Discrete integral equation, a={a}")
plt.xlabel("Dimension d")
plt.ylabel("Time (seconds)")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("rootfinding_timing_integral.png", dpi=300)
plt.show()
