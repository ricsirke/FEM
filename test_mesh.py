############################
#from mesh import *
from mesh_neu import *
############################

from Tkinter import *

magn = 400
offset = 30
offset_vec = Vector(offset,offset)

def draw_canvas(canvas, bound, color):
		rad = 3
		pos = bound.pos.coords
		coords = (pos[0]*magn - rad + offset, pos[1]*magn - rad + offset, pos[0]*magn + rad + offset, pos[1]*magn + rad + offset)
		canvas.create_oval(*coords, fill=color)
		
def draw_face(canvas, face):
    canvas.create_polygon((face.nodes[0].pos*magn + offset_vec).coords, (face.nodes[1].pos*magn + offset_vec).coords, (face.nodes[2].pos*magn + offset_vec).coords, fill="", outline="black")

	
def draw_mesh(m):
	root = Tk()
	canvas = Canvas(root, width=1000, height=700, bg="white")
		
	for face in m.faces:
		draw_face(canvas, face)
		
	for bound in m.boundaries:
		draw_canvas(canvas, bound, "green")
		
	for node in m.nodes:
		if node.neu:
			draw_canvas(canvas, node, "orange")
		else:
			draw_canvas(canvas, node, "red")
			
	node_list = []
	neu_node = [node for node in m.nodes if node.neu][6]
	#neu_neighs = [node for node in neu_node.get_neighbour_nodes(withBounds=True)]
	neu_neighs = [neigh for neigh in neu_node.get_neighbour_nodes(withBounds=True) if neigh.pos.coords[1] == 0 and neigh.pos.coords[0] != neu_node.pos.coords[0]]
	
	draw_canvas(canvas, neu_node, "white")
	
	for node in neu_neighs:
		draw_canvas(canvas, node, "black")
		
	canvas.pack()
	root.mainloop()

	
	
def test_mesh(fine_times):
	mc = Mesh_control(fine_times)
	m = mc.make_mesh(x1=0.0, y1=0.0, x2=0.0, y2=1.0, x3=2.0, y3=0.0)	
	
	draw_mesh(m)		
	
      
      
fine_times = int(raw_input("Specify number of grid refinements!\n"))
test_mesh(fine_times)