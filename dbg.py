


def g(v):
    return v[0]

def ter(i):
    return (1.0/4.0)**i
    

FIN = 4.0
ti = 1.0/15.0
Ni = [1.0,0.0]
    

def b(i=42.0):    
    return (ter(FIN))/3.0 - (g(Ni)*ti)/2.0
    
    

    
    

print b()