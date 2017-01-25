import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numpy.linalg import solve, det
from mesh_neu import *
from compgeom import *
from pylab import *
import sys

def plane_val(point, plane_coeff):
    return np.dot(plane_coeff, point)
    
def getNodeByPosCoords(nodes, pos_coords):
    for ndind in range(len(nodes)):
        nd = nodes[ndind]
        #print "ndcoords:", nd.pos.coords, "poscoord:", pos_coords
        if (nd.pos.coords[0] == pos_coords[0]) and (nd.pos.coords[1] == pos_coords[1]):
            #print "in"
            return nodes[ndind], ndind
    Exception('getNodeByPosCoords')
    
def setVertsNeu(mesh, vert_pos):
    nodes = mesh.boundaries
    
    nd, ndi = getNodeByPosCoords(nodes, vert_pos)
    print "ndi:", ndi
    
    #mesh.nodes[ndi]
    mesh.nodes[ndi].neu = True
    


def load_function(x,y):
    return 1
    
def pont_sol(x,y):
    return x*(1-(x/2)-y)


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

    

    
def run_plot(m):    
    tri_area = m.get_one_tria_area()
    print "tri_area:", tri_area
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
    
    small_side = m.get_tria_smallest_side()
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
        
        #####################################################
        
        # constructing b
        # ith comp = 1/3 * f(Ni) * (6*area of one tria)
        node_coords = m.nodes[i].pos.coords
        conn_faces = m.nodes[i].faces
        bound_integral = 0
        
        if m.nodes[i].neu:
            #neu_neighs = [neigh for neigh in m.nodes[i].get_neighbour_nodes(withBounds=True) if neigh.pos.coords[1] == 0 and neigh.pos.coords[0] != node_coords[0]]
            #print len(neu_neighs), "csucs", m.nodes[i].pos.coords, "szomszedok", neu_neighs[0].pos.coords, neu_neighs[1].pos.coords
            #x_coords = [neu_neighs[0].pos.coords[0], neu_neighs[1].pos.coords[0]]
            #bound_integral = bound_int(min(x_coords), node_coords[0], max(x_coords))
            
            bound_integral = (bound_g(*node_coords)*(2/16.0))/2.0
            print "ind:", i, "boundint:", bound_integral
            
        if m.nodes[i].isBoundary:
            pass
            
            
            #print bound_g(*node_coords), bound_integral
        #b[i] = (load_function(*node_coords) * (len(conn_faces)*tri_area))/3 + bound_integral
        b[i] = (tri_area)/3.0 + bound_integral
        
    print A    
    print "determinant of the stiffness matrix: " + str(det(A))
    
    #####################################################

    # solving Ax=b
    x = solve(A,b)
    
    ########################################################################################################
    
    #DBG
    
    import dbg
    
    # neupos
    neucount = 0
    for ndi in range(len(m.nodes)):
        nd = m.nodes[ndi]
        if nd.neu:
            neucount += 1
            print ndi, nd.pos.coords
    print "# of neu nodes:", neucount
    
    print "\n\ntri area:", tri_area
    #print "N_0 pos:", m.nodes[0].pos.coords
    
    dbg.test_b()
    print "computed b values"
    print "b[0]:", b[0]
    print "b[50]:", b[50]
    print "b[53]:", b[53]
    
    print "\n"
    nhs1 = m.nodes[50].get_neighbour_nodes()
    nhs2 = m.nodes[24].get_neighbour_nodes()
    missings = [ nod for nod in nhs1 if nod in nhs2 ]
    miss = missings[0].pos.coords
    

    print "missing:", missings
    print "miss:", getNodeByPosCoords(m.nodes, miss)
    
    print "A[50][24]:", A[50][24]
    
    
    print "test_A:", dbg.test_A(m.nodes)
    
    print x
    
    ########################################################################################################
    
    # error calc

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
    
    ########################################################################################################
    
    # plotting the solution
    all_nodes = m.nodes + m.boundaries
    all_values = np.concatenate((x, [0 for i in range(len(m.boundaries))]))
    sol_plot(all_nodes, all_values)

    
    
def run(pointlist=[]):
    m = load_mesh(*pointlist)
    print "mesh is ready with", len(m.faces), "faces and", len(m.nodes), "nodes"
    run_plot(m)    
        
        
run()