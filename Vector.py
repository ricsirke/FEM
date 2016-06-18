from math import sqrt

class Vector():
    def __init__(self, *args):
        # Vector(2,3,4,3,2)
        self.coords = []
        
        for arg in args:
            self.coords.append(arg)      
        
    def set(self, *args):
        if len(self.coords) != len(args):
            print 'Error in Vector: set method, different dimensions'
        else:
            self.coords = args
    
    def inccoord(self, place, coord):
        self.coords[place] += coord
        
    def get(self):
        return self.coords
    
    def getCoord(self, a):
        return self.coords[a]
    
    def len(self):
        """sum = 0
        for coord in self.coords:
            sum += coord**2
        return sqrt(sum)"""
        max = 0
        for coord in self.coords:
            if max < coord:
                max = coord
        return max 
		
	def llen(self):
		sum = 0
        for coord in self.coords:
            sum += coord**2
        return sqrt(sum)
    
    def __add__(self, b):
        return Vector(*[self.coords[i] + b.coords[i] for i in range(len(self.coords))])
        
    def __sub__(self, b):
        return Vector(*[self.coords[i] - b.coords[i] for i in range(len(self.coords))])
    
    def __div__ (self, c):
        if isinstance(c, (long, int, float)):
            return Vector(*[self.coords[i] / c for i in range(len(self.coords))])
        else:
            print "Error in Vector - div: second arg is not i can div with"
        
    def __mul__ (self, c):
        if isinstance(c, (long, int, float)):
            return Vector(*[self.coords[i] * c for i in range(len(self.coords))])
        elif isinstance(c, Vector):
            return Vector(*[self.coords[i] * c.coords[i] for i in range(len(self.coords))])
        else:
            print "Error in Vector - div: second arg is not i can multiply with"
            
    def __repr__(self):
        mystr = "Vector("
        for i in range(len(self.coords)-1):
            mystr += str(self.coords[i]) + ", "
        mystr += str(self.coords[len(self.coords)-1]) + ")"
        return mystr