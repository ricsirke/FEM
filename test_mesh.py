from mesh import *
from Tkinter import *

def create_mesh(fine_times):
	m = Mesh()
	m.boundaries.append(Node(Vector(0.2,0.2)))
	m.boundaries.append(Node(Vector(0.2,1.2)))
	m.boundaries.append(Node(Vector(2.2,0.2)))
	m.faces.append(m.make_tri(m.boundaries[0],m.boundaries[1],m.boundaries[2]))
	
	for i in range(fine_times):
		m.fine_mesh()
	return m

def test_mesh_bounds(fine_times):
	m = create_mesh(fine_times)
	
	root = Tk()
	canvas = Canvas(root, width=1000, height=700, bg="white")
	magn = 400
	
	def draw_canvas(bound, color):
		rad = 4
		pos = bound.pos.coords
		coords = (pos[0]*magn - rad, pos[1]*magn - rad, pos[0]*magn + rad, pos[1]*magn + rad)
		canvas.create_oval(*coords, fill=color)
		
	def draw_face(face):
		canvas.create_polygon((face.nodes[0].pos*magn).coords, (face.nodes[1].pos*magn).coords, (face.nodes[2].pos*magn).coords, fill="", outline="black")
	
	for face in m.faces:
		draw_face(face)
		
	for bound in m.boundaries:
		draw_canvas(bound, "green")
		
	for node in m.nodes:
		draw_canvas(node, "red")
		
	canvas.pack()
	root.mainloop()
		

def test_smallest_side(fine_times):
    m = create_mesh(fine_times)
    small_side = m.get_tria_smallest_side()
    
    print "the smallest side: ", small_side
        
#test_mesh_bounds(4)