from example_integral_equation import *
from pylab import *
from BroydenMatrix import BroydenM

N = 300
a = 2
tol = 1e-13
i_max = 8

t_newton              = np.zeros(i_max)
t_broyden_good_full   = np.zeros(i_max)
t_broyden_good_linop  = np.zeros(i_max)
t_broyden_good_simple = np.zeros(i_max)
t_broyden_good_svd    = np.zeros(i_max)

dd = 20 * 2**arange(i_max)

for i in range(i_max):
    d = dd[i]
    print("Dimension d = ", d)
    f, J = setup_integral_equation(d, a=a)
    
    t = time.process_time()
    ### Your computation here
    t_newton[i] = time.process_time()-t

    t = time.process_time()
    ### Your computation here
    t_broyden_good_full[i] = time.process_time()-t

    t = time.process_time()
    ### Your computation here
    t_broyden_good_linop[i] = time.process_time()-t
    
    t = time.process_time()
    ### Your computation here
    t_broyden_good_simple[i] = time.process_time()-t
    
    t = time.process_time()
    ### Your computation here
    t_broyden_good_svd[i] = time.process_time()-t

figure()
### Plotting here
title(f"Discrete integral equation, $a={a}$")
legend()
xlabel("Dimension $d$")
ylabel("Time")

show()
