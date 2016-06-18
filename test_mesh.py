from mesh import *
from Tkinter import *

def create_mesh():
	m = Mesh()
	m.add_node(Node(Vector(0.2,0.2)))
	m.add_node(Node(Vector(0.2,1.2)))
	m.add_node(Node(Vector(2.2,0.2)))
	m.faces.append(m.make_tri(m.nodes[0],m.nodes[1],m.nodes[2]))
	m.fine_mesh()
	m.fine_mesh()
	m.fine_mesh()
	#m.fine_mesh()
	return m

def test_mesh_bounds():
	m = create_mesh()
	
	root = Tk()
	canvas = Canvas(root, width=500, height=500, bg="white")
	
	def draw_canvas(bound):
		coords = ()
		canvas.create_oval(*coords, fill="black")
		
	for bound in m.boundaries:
		draw_canvas(bound)
		
	canvas.pack()
	root.mainloop()
		
	

def test_mesh_inner():
	m = create_mesh()	

	root = Tk()
	canvas = Canvas(root, width=500, height=500, bg="white")
	def draw_canvas(face):
		canvas.create_polygon((face.nodes[0].pos*200).coords, (face.nodes[1].pos*200).coords, (face.nodes[2].pos*200).coords, fill="", outline="black")
	"""def fine():
		m.fine_mesh()
		#canvas.delete("all")
		draw_canvas()
	btn_fine = Button(root, command=fine, text="Fine")
	btn_fine.pack()"""
	for face in m.faces:
		draw_canvas(face)
	canvas.pack()
	root.mainloop()
	
test_mesh()