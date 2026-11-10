from BroydenMatrix import BroydenM
from numpy.random import rand
from numpy import eye, zeros, ones, allclose, newaxis, arange

def unitvector(n, i):
    v = zeros(n)
    v[i] = 1
    return v
    
def test_broyden_matrix_4():
    A = BroydenM(4)
    A.append(rand(4), rand(4))
    A.append(unitvector(4,1), unitvector(4,0))
    A.append(unitvector(4,2), unitvector(4,2))
    A.append(unitvector(4,0), unitvector(4,3))
    A.prune(3, method='Simple')
    assert allclose(A.todense(),
                    [[1,0,0,1],[1,1,0,0],[0,0,2,0],[0,0,0,1]])

def test_broyden_matrix_5():
    A = BroydenM(4)
    u = arange(4)
    v = arange(3,7)
    w = ones(4)
    A.append(u,unitvector(4,0))
    A.append(unitvector(4,2), v)
    A.append(v,unitvector(4,1))
    A.append(w,u)
    A.append(unitvector(4,3),w)
    A.prune(2, method='SVD')
    Aref = [[2.4207836712495254, 3.3341343885599306, 2.2155139032378064, 2.9049858162302287], [1.9908790236368812, 5.536596248836144, 2.175909847423853, 2.9130787263410705], [4.380565209438925, 10.290137955559848, 7.9016910437125185, 9.044932666973082], [3.649887265675505, 8.163748499085107, 2.9381446467213417, 5.030470492776266]]
    assert allclose(A.todense(), Aref)
