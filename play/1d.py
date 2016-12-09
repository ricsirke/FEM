# -*- coding: utf-8 -*-

# u : [0, 1] -> R
# u(0) = 0
# u(1) = 0
# f(x) = -2

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
print "\n"

nx = 10
xmin = 0.0
xmax = 1.0

h = (xmax-xmin)/nx

nodes = np.array([xmin + k*h for k in range(nx+1)])[1:-1]
print "nodes:", nodes, "\n"

def source_fn(x):
    return 2.0
    
f = [source_fn(nd) for nd in nodes]
    
def u_pont(x):
    return -(x**2) + x

u_pont = np.array([ u_pont(nd) for nd in nodes ])

grPhi1 = nx
grPhi2 = -nx

A = np.zeros((nx-1,nx-1))

# looping thrugh the elements
A[0][0] += grPhi1*grPhi1*h
A[-1][-1] += grPhi2*grPhi2*h

for i in range(1, nx-1):
    A[i][i] += grPhi2*grPhi2*h
    A[i][i-1] += grPhi1*grPhi2*h
    A[i-1][i-1] += grPhi1*grPhi1*h
    A[i-1][i] += grPhi2*grPhi1*h
    
#print A
print "A:\n", A, "\n"

# b
b = np.array([ h*f[i] for i in range(nx-1) ])
print "b:\n", b, "\n"
    

# Ax = b
u = np.linalg.solve(A, b)

print "solution:\n", u, "\n"
print "pont_sol:\n", u_pont, "\n"


# errors
errors = np.array([ np.abs(u[i] - u_pont[i]) for i in range(len(nodes)) ])
print "errors:", errors, "\n"
print "error sum:", sum(errors)


plt.plot(u, 'r')
plt.show()