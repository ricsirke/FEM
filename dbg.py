
    
    
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