import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numpy.linalg import solve, det
from mesh import *
from compgeom import *
from pylab import *
import sys

def load_function(x,y):
    return 2*x + y
    
def pont_sol(x,y):
    return x*y*(1-(x/2)-y)
    
# get plane val
def plane_val(point, plane_coeff):
    return np.dot(plane_coeff, point)
    
def getNodeByPosCoords(nodes, pos_coords):
    for ndind in range(len(nodes)):
        if nodes[ndind].pos.coords == pos_coords:
            return nodes[ndind], ndind
    return None, None
    
    


def load_mesh():
    mc = Mesh_control(int(sys.argv[1]))
    mc.make_mesh(x1=0.0, y1=0.0, x2=0.0, y2=1.0, x3=2.0, y3=0.0)
    return mc.mesh
    
def sol_plot(point_coords, point_values):
    # plotting the solution
    xi = np.linspace(0, 2, 1000)
    yi = np.linspace(0, 2, 1000)
    x_ax = [node.pos.coords[0] for node in point_coords]
    y_ax = [node.pos.coords[1] for node in point_coords]

    zi = griddata(x_ax, y_ax, point_values, xi, yi, interp='linear')
    
    cs = plt.contourf(xi,yi,zi, cmap=plt.cm.rainbow, vmax=abs(zi).max(), vmin=-abs(zi).max())
    cbar = plt.colorbar(cs)
    plt.show()
    
    
def run_plot(m):
    tri_area = m.get_one_tria_area()
    print tri_area
    N = len(m.nodes)

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
        #face.Mmx = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                node_pair = (face.nodes[i], face.nodes[j])
                product = np.dot(all_grad[i], all_grad[j])*tri_area
                face.M[node_pair] = product
                face.M[tuple(reversed(node_pair))] = product
                #face.Mmx[i,j] = product
                
        #print face.Mmx
        #print "determinant of the element stiffness matrix: " + str(det(face.Mmx))
    print "element stiffness matrixes are ready"
        
    
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    ready_inds = {}
    
    for i in range(N):
        # constructing A
        
        neighbour_nodes = m.nodes[i].get_neighbour_nodes()

        for node in neighbour_nodes:
        # using symmetry
            try:
                isReady = ready_inds[(i,node.index)]
            except KeyError:
                    sum = 0
                    neighbour_faces = [face for face in node.faces if face in m.nodes[i].faces]

                    for face in neighbour_faces:
                        node_pair = (m.nodes[i], node)
                        try:
                            sum += face.M[node_pair]
                        except KeyError:
                            pass
                            
                    A[i,node.index] = sum
                    A[node.index, i] = sum
                    
                    ready_inds[(i,node.index)] = True
                    ready_inds[(node.index,i)] = True
            
        #print str(i)+"/"+str(N)
        
        # constructing b
        # ith comp = 1/3 * f(Ni) * (6*area of one tria)
        node_coords = m.nodes[i].pos.coords
        # how many triangle has this node
        conn_faces = m.nodes[i].faces
        #print len(conn_faces)
        b[i] = (load_function(*node_coords) * (len(conn_faces)*tri_area))/3
        
    #print A
    #print "determinant of the stiffness matrix: " + str(det(A))
    print "stiffness matrix, load vector ready"

    # solving Ax=b    
    x = solve(A,b)
    print "linear system solved"
    
    # error calc
    xpont = np.zeros(N)
    error = np.zeros(N)
    
    smallest_side = m.get_tria_smallest_side()
    print "smallest side ", smallest_side
    
    """
        errors = []
        pont = [0.125, 0.5]
        for i in range(N):
            node_coords = m.nodes[i].pos.coords
            xpont[i] = pont_sol(*node_coords)
            error[i] = np.fabs(xpont[i] - x[i])
            if node_coords == pont:
                print "pont, koz, hiba: ", xpont[i], x[i], error[i]
            errors.append({"error": error[i], "pos": m.nodes[i].pos, "i":i})
            
        error_pont = [err["error"] for err in errors if err["pos"].coords == pont]
        errors_num = [err["error"] for err in errors]
        print "error in ", pont, ": ", error_pont
        print "sum of errors: ", np.sum(errors_num)
    """
    
    # creating the middle point to check errors for the different resolutions
    err_point = [0,0]
    for v in m.mainVerts:
        for i in range(len(err_point)):
            err_point[i] += v[i]            
    err_point = map(lambda x:x/3, err_point)
    
    # which triangle contains the middle point?
    for face in m.faces:
        fcnds = face.nodes
        #print "fcnd", fcnds[0].pos.coords
        
        if p_inside_tria(err_point, fcnds):
            err_face = face
            break
    
    # how to know that for a certain point in plane by coords, what is the value of the solution there
    err_fcnds = err_face.nodes
    #print err_fcnds
    p1 = err_fcnds[0].pos.coords
    p2 = err_fcnds[1].pos.coords
    p3 = err_fcnds[2].pos.coords
    
    nd1, nd1i = getNodeByPosCoords(m.nodes, p1)
    nd2, nd2i = getNodeByPosCoords(m.nodes, p2)
    nd3, nd3i = getNodeByPosCoords(m.nodes, p3)
    
    err_A = [ [p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1] ]
    err_b = [ x[nd1i], x[nd2i], x[nd3i] ] # !!!!!!!!
    err_planecoeff = solve(err_A, err_b)
    #print "err_planecoeff:", err_planecoeff
    #print "err_point:", err_point
    sol_appr = plane_val(err_point + [1], err_planecoeff)
    sol_diff = np.abs(sol_appr - pont_sol(*err_point))
    print "error:", sol_diff
    
    # plotting the solution
    all_nodes = m.nodes + m.boundaries
    # mapping by index
    all_values = np.concatenate((x, [0 for i in range(len(m.boundaries))]))
    sol_plot(all_nodes, all_values)

    
    
def run(pointlist=[]):
    m = load_mesh(*pointlist)
    print "mesh is ready with", len(m.faces), "faces and", len(m.nodes), "nodes"
    run_plot(m)    
        
        
run()