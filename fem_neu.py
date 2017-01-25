import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numpy.linalg import solve, det
from mesh_neu import *
from pylab import *
import sys

def load_function(x,y):
    return 1
    
def pont_sol(x,y):
    return x*(1-(x/2)-y)
    
def bound_g(x,y):
    #return -1 + x + y
    return x
    
# primitive functions for the boundary integrals
def F1(x):
    return ((x**3)/3) - ((x**2)/2)
    
def F2(x):
    return ((x**3)/3) - (x**2) + x

# formula for the boundary integrals
def bound_int(x0, x1, x2):
    return F1(x1) - F1(x0) - F2(x2) + F2(x1)

    
    
def load_mesh():
    mc = Mesh_control(int(sys.argv[1]))
    mc.make_mesh(x1=0.0, y1=0.0, x2=0.0, y2=1.0, x3=2.0, y3=0.0)
    return mc.mesh
    
def run_plot(m):
    N = len(m.nodes)

    tri_area = m.get_one_tria_area()
    for face in m.faces:
        b0 = np.array([1.0,.0,.0])
        b1 = np.array([.0,1.0,.0])
        b2 = np.array([.0,.0,1.0])
        
        M = []
        for i in range(3):
            M.append(face.nodes[i].pos.coords + [1])
        M = np.array(M)
        #print M
        #print "determinant of the calc matrix: " + str(det(M))
        #print ""
        
        all_grad = [solve(M,b0)[:2], solve(M,b1)[:2], solve(M,b2)[:2]]
        face.M = {}
        face.Mmx = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                node_pair = (face.nodes[i], face.nodes[j])
                product = np.dot(all_grad[i], all_grad[j])*tri_area
                face.Mmx[i,j] = product
                
                face.M[node_pair] = product
                face.M[tuple(reversed(node_pair))] = product                                
                
        #print face.Mmx
        #print "determinant of the element stiffness matrix: " + str(det(face.Mmx))
        
    
    A = np.zeros((N, N))
    b = np.zeros(N)
        
    small_side = m.get_tria_smallest_side()
    
    for i in range(N):
        # constructing A
        
        # get neighbouring faces' nodes
        neighbour_nodes = m.nodes[i].get_neighbour_nodes()

        for node in neighbour_nodes:
            sum = 0    
            neighbour_faces = [face for face in m.nodes[i].faces if face in node.faces]
            for face in neighbour_faces:
                node_pair = (m.nodes[i], node)
                if node_pair in face.M.keys():
                    sum += face.M[node_pair]
            A[i,node.index] = sum
        #print str(i)+"/"+str(N)
        
        #####################################################
        
        # constructing b
        # ith comp = 1/3 * f(Ni) * (6*area of one tria)
        node_coords = m.nodes[i].pos.coords
        conn_faces = m.nodes[i].faces
        bound_integral = 0
        
        if m.nodes[i].neu:
            neu_neighs = [neigh for neigh in m.nodes[i].get_neighbour_nodes(withBounds=True) if neigh.pos.coords[1] == 0 and neigh.pos.coords[0] != node_coords[0]]
            
            #print len(neu_neighs), "csucs", m.nodes[i].pos.coords, "szomszedok", neu_neighs[0].pos.coords, neu_neighs[1].pos.coords
                
            x_coords = [neu_neighs[0].pos.coords[0], neu_neighs[1].pos.coords[0]]
            bound_integral = bound_int(min(x_coords), node_coords[0], max(x_coords))
            #print bound_integral
            
        if m.nodes[i].isBoundary:
            pass
            
            
            #print bound_g(*node_coords), bound_integral
        #b[i] = (load_function(*node_coords) * (len(conn_faces)*tri_area))/3 + bound_integral
        b[i] = (tri_area)/3 - (bound_g(*node_coords)*(1/15.0))/2
        
    #print A
    #print "determinant of the stiffness matrix: " + str(det(A))
    
    #####################################################

    # solving Ax=b
    x = solve(A,b)
    
    ########################################################################################################
    ########################################################################################################
    
    # error calc
    xpont = np.zeros(N)
    error = np.zeros(N)
    
    smallest_side = m.get_tria_smallest_side()
    
    errors = []
    for i in range(N):
        node_coords = m.nodes[i].pos.coords
        xpont[i] = pont_sol(*node_coords)
        error[i] = np.absolute(xpont[i] - x[i])
        if error[i] > 0.1:
            errors.append({"error": error[i], "pos": m.nodes[i].pos, "i":i})
            
    #errors_num = [err["error"] for err in errors if err["pos"].coords == [0.5, 0.5]]
    errors_num = [err["error"] for err in errors]
    #print "error in (0.5, 0.5): ", max(errors_num)
    print "sum of errors: ", np.sum(errors_num)
    
    ########################################################################################################
    
    #DBG
    # neupos
    neucount = 0
    for ndi in range(len(m.nodes)):
        nd = m.nodes[ndi]
        if nd.neu:
            neucount += 1
            #print ndi, nd.pos.coords
    print "# of neu nodes:", neucount
    
    print "\n b[0]:", b[0]
    
    ########################################################################################################
        
    # plotting the solution
    xi = np.linspace(0, 2, 1000)
    yi = np.linspace(0, 3, 1000)
    x_ax = [node.pos.coords[0] for node in m.nodes]
    y_ax = [node.pos.coords[1] for node in m.nodes]

    zi = griddata(x_ax, y_ax, x, xi, yi, interp='linear')
    cs = plt.contourf(xi,yi,zi, cmap=plt.cm.rainbow, vmax=abs(zi).max(), vmin=-abs(zi).max())
    cbar = plt.colorbar(cs)
    plt.show()

    
    
def run(pointlist=[]):
    m = load_mesh(*pointlist)
    print "mesh is ready"
    run_plot(m)    
        
        
run()