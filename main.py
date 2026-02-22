import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # needed for 3D plotting

# Load data: each line has   x   y   u(x,y)
data = np.loadtxt("solution.dat")

# Extract columns
x = data[:,0]
y = data[:,1]
u = data[:,2]

# Determine grid size
# Because x and y come from a uniform (m-1)x(m-1) mesh
# we find unique values
nx = len(np.unique(x))
ny = len(np.unique(y))

# Reshape into 2D mesh
X = x.reshape(nx, ny)
Y = y.reshape(nx, ny)
U = u.reshape(nx, ny)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, cmap='viridis', edgecolor='none')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y)")
ax.set_title("Solution of Poisson Equation")

plt.show()
