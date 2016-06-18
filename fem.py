import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numpy.linalg import solve, det
from mesh import *
from pylab import *

def load_function(x,y):
	return 2*x + y
	
def pont_sol(x,y):
	return x*y*(1-(x/2)-y)


def load_mesh():
	mc = Mesh_control()
	mc.make_mesh(x1=0.0, y1=0.0, x2=0.0, y2=1.0, x3=2.0, y3=0.0)
	return mc.mesh
	
def run_plot(m):
	tri_area = m.get_one_tria_area()
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
		face.Mmx = np.zeros((3,3))
		for i in range(3):
			for j in range(3):
				node_pair = (face.nodes[i], face.nodes[j])
				product = np.dot(all_grad[i], all_grad[j])*tri_area
				face.M[node_pair] = product
				face.M[tuple(reversed(node_pair))] = product
				face.Mmx[i,j] = product
				
		#print face.Mmx
		#print "determinant of the element stiffness matrix: " + str(det(face.Mmx))

		
	
	A = np.zeros((N, N))
	b = np.zeros(N)
	
	
	for i in range(N):
		# constructing A
		
		# get neighbouring faces' nodes
		neighbour_nodes = m.nodes[i].get_neighbour_nodes()

		for node in neighbour_nodes:
			sum = 0
			neighbour_faces = list(set([face for face in m.nodes[i].faces]).intersection(set([face for face in node.faces])))
			for face in neighbour_faces:
				node_pair = (m.nodes[i], node)
				if node_pair in face.M.keys():
					sum += face.M[node_pair]
			A[i,node.index] = sum
		#print str(i)+"/"+str(N)
		
		# constructing b
		# ith comp = 1/3 * f(Ni) * (6*area of one tria)
		node_coords = m.nodes[i].pos.coords
		# how many triangle has this node
		conn_faces = [face for face in m.faces if m.nodes[i] in face.nodes]
		#print len(conn_faces)
		b[i] = (load_function(*node_coords) * (len(conn_faces)*tri_area))/3
		
	#print A
	#print "determinant of the stiffness matrix: " + str(det(A[1:-1,1:-1]))

	#with open('m.txt', 'w') as f:
	#	f.write(str(A))


	# solving Ax=b	
	x = solve(A,b)
	#print x
	
	# error calc
	xpont = np.zeros(N)
	error = np.zeros(N)
	
	smallest_side = m.get_tria_smallest_side()
	print smallest_side
	
	errors = []
	for i in range(N):
		node_coords = m.nodes[i].pos.coords
		xpont[i] = pont_sol(*node_coords)
		error[i] = abs(xpont[i] - x[i])
		errors.append({"error": error[i], "pos": m.nodes[i].pos, "i":i})
		
	#errors_num = [err["error"] for err in errors if err["pos"].coords == [0.5, 0.5]]
	errors_num = [err["error"] for err in errors]
	#print "error in (0.5, 0.5): ", max(errors_num)
	print "sum of errors: ", np.sum(errors_num)
		
	# plotting the solution
	xi = np.linspace(0, 2, 1000)
	yi = np.linspace(0, 2, 1000)
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