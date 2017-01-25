from Vector import Vector
import itertools as it
from numpy import absolute
from numpy import array as nparray
from numpy.linalg import norm
from math import sqrt
#from scipy.spatial import Delaunay

class Node():
    def __init__(self, pos_vector):
        self.pos = pos_vector
        self.faces = []
        self.isBoundary = False
        self.index = None
        self.neu = False
        
    def get_neighbour_nodes(self, withBounds=False):
        """not returning boundaries"""
        neighbour_nodes = []
        for face in self.faces:
            for a_node in face.nodes:
                if a_node not in neighbour_nodes and ((not a_node.isBoundary) or (withBounds)):
                    neighbour_nodes.append(a_node)
        return neighbour_nodes
        
    def get_neighbour_faces(self):
        neighbour_faces = []
        for face in self.faces:
            if face not in neighbour_faces:
                neighbour_faces.append(face)
        return neighbour_faces
    
class Face():    
    def __init__(self, n1, n2, n3):
        self.M = None
        self.nodes = []
        self.nodes.append(n1)
        self.nodes.append(n2)
        self.nodes.append(n3)
        
class Mesh():
    def __init__(self):
        self.nodes = []
        self.faces = []
        self.boundaries = []
    
    def get_one_tria(self):
        n1 = self.nodes[4]
        n1_neigh = n1.get_neighbour_nodes()
        n2 = n1_neigh[0]
        n2_neigh = n2.get_neighbour_nodes()
        n1_n2_intersect = [item for item in n1_neigh if item not in n2_neigh]
        n3 = list(n1_n2_intersect)[0]
        return [n1,n2,n3]
        
    def get_one_tria_area(self):
        tria = self.get_one_tria()
        return absolute((tria[0].pos.coords[0] - tria[2].pos.coords[0])*(tria[1].pos.coords[1] - tria[0].pos.coords[1]) - (tria[0].pos.coords[0] - tria[1].pos.coords[0])*(tria[2].pos.coords[1] - tria[0].pos.coords[1]))/2
    
    def get_tria_smallest_side(self):
        tria = self.get_one_tria()
        s1 = nparray((tria[0].pos - tria[1].pos).coords)
        s2 = nparray((tria[0].pos - tria[2].pos).coords)
        s3 = nparray((tria[1].pos - tria[2].pos).coords)
        
        lens = [norm(s1, ord=2), norm(s2, ord=2), norm(s3, ord=2)]
        return min(lens)
    
    def set_indices(self):
        for i in range(len(self.nodes)):
            self.nodes[i].index = i
    
    def add_node(self, node):
        self.nodes.append(node)
    
    def make_tri(self, n1, n2, n3):
        new_face = Face(n1,n2,n3)
        
        n1.faces.append(new_face)
        n2.faces.append(new_face)
        n3.faces.append(new_face)
        
        return new_face
    
    def fine_mesh(self):
        def will_be_boundary(mid_node, parent_nodes):
            # have to watch parent nodes common neighbours - can be a subroutine
            n1_neighbours = []
            n2_neighbours = []
            
            for face in parent_nodes[0].faces:
                n1_neighbours += face.nodes
            for face in parent_nodes[1].faces:
                n2_neighbours += face.nodes
            
            # boundary if and only if the parent nodes has 0 or 1 common neighbours
            # WHY?????
            intersect = [nde for nde in n1_neighbours if nde in n2_neighbours]
            if len(intersect) == 0 or len(intersect) == 1:
                return True
            else:
                return False
                
        def will_be_neu_boundary(mid_node, parent_nodes):
            if parent_nodes[0].neu and parent_nodes[1].neu and will_be_boundary(mid_node, parent_nodes):
                return True
            else:
                return False
        
        def is_ready_node_pair(node_pair):
            try:
                node_pair_to_mid[node_pair]
                return True
            except KeyError:
                pass
            try:
                node_pair_to_mid[node_pair[::-1]]
                return True
            except KeyError:
                return False
                

        ready_edges = []
        new_faces = []
        mid_to_node_pair = {}
        node_pair_to_mid = {}
        
        while self.faces != []:
            # delete a face
            face = self.faces.pop()
            
            for node in face.nodes:
                # refresh corresponding nodes, deleting the appropriate face from them
                for i in range(len(node.faces)):
                    if node.faces[i] == face:
                        node.faces.pop(i)
                        break
                        
            # contruct new nodes !!!!if we didn't construct earlier!!!!
            new_nodes = []
            for node_pair in list(it.combinations(face.nodes, 2) ):
                # if we didn't construct it earlier
                # if constructed, we have to have it for the new faces                 node_pair -> mid_node     mid_node -> node_pair
                if not is_ready_node_pair(node_pair):
                    new_node = Node( node_pair[0].pos + (node_pair[1].pos - node_pair[0].pos)/2 )
                    if will_be_boundary(new_node, node_pair) and not will_be_neu_boundary(new_node, node_pair):
                        new_node.isBoundary = True
                        self.boundaries.append( new_node )
                    elif will_be_neu_boundary(new_node, node_pair):
                        new_node.neu = True
                        self.nodes.append(new_node)
                    else:
                        self.nodes.append( new_node )
                        
                    new_nodes.append( new_node )
                    node_pair_to_mid[node_pair] = new_node
                    mid_to_node_pair[new_node] = node_pair
                else:
                    # have to append for new faces
                    if node_pair in node_pair_to_mid.keys():
                        new_nodes.append(node_pair_to_mid[node_pair])
                    else:
                        new_nodes.append(node_pair_to_mid[node_pair[::-1]])

            # construct new faces
            new_faces.append(self.make_tri(*new_nodes))
            for new_node_pair in list(it.combinations(new_nodes, 2)):
                # need the 3rd, closest, old node        intersection of new_node_pair's nodes' old_node_pairs            
                good_old_node = set(mid_to_node_pair[new_node_pair[0]]).intersection(set(mid_to_node_pair[new_node_pair[1]]))
                new_faces.append(self.make_tri(new_node_pair[0], new_node_pair[1], *good_old_node))
                        
            #print len(self.faces)
            
        for new_face in new_faces:
            self.faces.append(new_face)
            
        self.set_indices()
    
class Mesh_control():
    def __init__(self, fine_times=4):
        self.mesh = None
        self.fine_times = fine_times        
            
    def make_mesh(self, x1=0.0, y1=0.0, x2=0.0, y2=1.0, x3=2.0, y3=0.0):
        self.mesh = Mesh()
        self.mesh.mainVerts = [(x1,y1),(x2,y2),(x3,y3)]
        
        n1 = Node(Vector(x1, y1))
        n1.neu = True
        n2 = Node(Vector(x2, y2))
        n3 = Node(Vector(x3, y3))
        n3.neu = True
        
        self.mesh.boundaries.append(n1)
        self.mesh.boundaries.append(n2)
        self.mesh.boundaries.append(n3)
        self.mesh.faces.append(self.mesh.make_tri(n1, n2, n3))
        
        for i in range(self.fine_times):
            print str(i) + "/" + str(self.fine_times), "fine"
            self.mesh.fine_mesh()
            
        return self.mesh
    
#mc = Mesh_control()
#mc.make_mesh()
#neu_nodes = [node for node in mc.mesh.nodes if node.neu == True]
#bou_nodes = mc.mesh.boundaries
#print neu_nodes
#print "neus", len(neu_nodes)
#print "boundaries", len(bou_nodes)
#mc.save_mesh("mesh.mf")
 
#import pickle
#print pickle.dumps(mesh.nodes[1])
    
    
from Tkinter import *
def test_mesh():
    
    m = Mesh()
    m.add_node(Node(Vector(0.2,0.2)))
    m.add_node(Node(Vector(0.2,1.2)))
    m.add_node(Node(Vector(2.2,0.2)))
    m.faces.append(m.make_tri(m.nodes[0],m.nodes[1],m.nodes[2]))
    m.fine_mesh()
    m.fine_mesh()
    #m.fine_mesh()
    #m.fine_mesh()

    

    root = Tk()
    canvas = Canvas(root, width=500, height=500, bg="white")
    def draw_canvas():
        canvas.create_polygon((face.nodes[0].pos*200).coords, (face.nodes[1].pos*200).coords, (face.nodes[2].pos*200).coords, fill="", outline="black")
    def fine():
        m.fine_mesh()
        #canvas.delete("all")
        draw_canvas()
    btn_fine = Button(root, command=fine, text="Fine")
    btn_fine.pack()
    for face in m.faces:
        draw_canvas()
    canvas.pack()
    root.mainloop()