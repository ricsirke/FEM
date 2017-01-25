import numpy as np
from numpy.linalg import solve

def test_A(nodes):
    FIN = 4.0 
    
    def ter(i):
        return (1.0/4.0)**i
    
    p50 = nodes[50].pos.coords
    p24 = nodes[24].pos.coords
    p49 = nodes[49].pos.coords
    A_ell = [ [ p50[0], p50[1], 1 ], [ p24[0], p24[1], 1 ], [ p49[0], p49[1], 1 ] ]
    b_ell_1 = [1, 0, 0]
    b_ell_2 = [0, 1, 0]
    
    grad50 = solve(A_ell, b_ell_1)[0:-1]
    grad24 = solve(A_ell, b_ell_2)[0:-1]
    
    return np.dot(grad50, grad24)*ter(FIN)
    
    
    
    
    
def test_b():
    FIN = 4.0 

    def g(v):
        return v[0]

    def ter(i):
        return (1.0/4.0)**i
        
    def hossz(i):
        return 2*((1.0/2.0)**i)

    def b(Ni):    
        return (ter(FIN))/3.0 - (g(Ni)*hossz(FIN))/2.0

    print "actual b values:"
    print "b0:", b([1.0,0.0])
    print "b50:", b([1.875, 0])
    print "b53:", b([1.625, 0])
    
    
    
if __name__ == "__main__":
    test_b()
    #test_A()