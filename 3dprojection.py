import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x= np.random.random(100)
y= np.random.random(100)
z= np.sin(3 * x**2 + y)

fig= plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z)

ax.plot(x, z, 'r+', zdir='y', zs=1.5)
ax.plot(y, z, 'g+', zdir='x', zs=-0.5)
ax.plot(x, y, 'k+', zdir='z', zs=-1.5)

ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 1.5])
ax.set_zlim([-1.5, 1.5])

plt.show() 
ax.set_zlim([-3, 3])

def projection3D(R,X,T):
	fig= plt.figure()
	ax = Axes3D(fig)

	proj2d = np.add(np.dot(R,X),T)

	x = proj2d[:,0]
	y = proj2d[:,1]

	ax.scatter(x,y)
	ax.plot(x, y, 'r+', zdir='z', zs=-1.5)

	plt.show()