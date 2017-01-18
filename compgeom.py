import numpy.linalg as la

def p_inside_tria(p, tria_verts):
    # use the explicit solution of the system
    p0 = tria_verts[0].pos.coords
    p1 = tria_verts[1].pos.coords
    p2 = tria_verts[2].pos.coords
    
    A = [[p1[0]-p0[0], p2[0]-p0[0]],[p1[1]-p0[1], p2[1]-p0[1]]]
    b = [p[0]-p0[0], p[1]-p0[1]]
    
    x = la.solve(A,b)
    s = x[0]
    t = x[1]
    
    return s < 1 and s > 0 and t < 1 and t > 0 and s + t < 1
    
    
if __name__ == "__main__":
    print p_inside_tria( [0.0,0.0], [ [0.0,0.0], [2.0,0.0], [0.0,1.0] ] )