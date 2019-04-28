import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

"""
This code demonstrates a single Bayesian update when using multivariate Gaussian
distributions. The notation is:

p(theta) 		~ N(theta0, sigma0)
p(x | theta) 	~ N(theta, sigma)

and we would like to compute p(theta | x) ~ p(x | theta)*p(theta)
"""

# Mean and covariance of the prior p(theta)
theta0 = np.array([[0], [0]])
sigma0 = np.array([[1, 0], [0, 1]])

# Mean and covariance of the likelihood function p(x | theta)
theta = np.array([[0], [0]])
sigma = np.array([[1, 0], [0, 1]])

# Generate fake observations drawn from a multivariate normal
# with mean at [10,10] and covariance Identity
N = 5000
mean = [10,10]
cov = [[1,0],[0,1]]
x1, x2 = np.random.multivariate_normal(mean, cov, N).T


thetaN = theta0

# Used for recording and plotting
sigmas = [sigma0]
thetas = [theta0]

# Perform the Bayesian update with each data point
for i in range(N):
	x = np.array([[x1[i]],[x2[i]]])
	sigmaN = np.linalg.inv(np.linalg.inv(sigma) + np.linalg.inv(sigma0))
	inner = np.linalg.inv(sigma).dot(x) + np.linalg.inv(sigma0).dot(thetaN)
	thetaN = sigmaN.dot(inner)

	# Record values
	sigmas.append(sigmaN)
	thetas.append(thetaN)
	
print "Finished Bayesian update:"
print "p(theta | x) ~ N(thetaN, sigmaN) where "
print "--> thetaN: ", thetaN
print "--> sigmaN: ", sigmaN

#Create grid and multivariate normal
x = np.linspace(-20,20,500)
y = np.linspace(-20,20,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
Z = bivariate_normal(X,Y,sigmas[0][0],sigmas[1][0],thetas[0][0],thetas[0][1])

# Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

