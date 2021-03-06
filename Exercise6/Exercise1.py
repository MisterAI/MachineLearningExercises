import matplotlib.pyplot as plt
import scipy as sp
import os
import numpy as np
import scipy.sparse.linalg as sla;
import pylab
import math

###############################################################################
# Helper functions
def mdot(*args):
	"""Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
	return reduce(np.dot, args)
def prepend_one(X):
	"""prepend a one vector to X."""
	return np.column_stack([np.ones(X.shape[0]), X])
def grid2d(start, end, num=50):
	"""Create an 2D array where each row is a 2D coordinate.
	np.meshgrid is pretty annoying!
	"""
	dom = np.linspace(start, end, num)
	X0, X1 = np.meshgrid(dom, dom)
	return np.column_stack([X0.flatten(), X1.flatten()])

def divideSquare(N):
    if N<1:
        return 1,1

    minX = max(1,int(np.floor(np.sqrt(N))))
    maxX = max(minX+1, int(np.ceil(np.sqrt(5.0/3*N))))
    Nx = range(minX, maxX+1)
    Ny = [np.ceil(N/y) for y in Nx]
    err = [np.uint8(y*x - N) for y in Nx for x in Ny]
    ind = np.argmin(err)
    y = Nx[int(ind/len(Ny))]
    x = Ny[ind%len(Nx)]
    return min(y,x), max(y,x)

############################################################################

def assign_to_clusters(X, mi_k, r_nk):
	# Assign the images to the clusters
	overall_error = 0
	i = 0
	j = 0
	min_k = 0
	for image in X:
		err_min = float("inf")
		min_k = 0
		j = 0
		for center in mi_k:
			squared_diff = mdot((image - center).T, image - center)
			if err_min > squared_diff:
				err_min = squared_diff
				min_k = j
			j += 1
		overall_error += err_min
		r_nk[min_k, i] = 1
		i += 1
	return overall_error

def adapt_means(mi_k,r_nk,X, K):
	for i in xrange(0, K):
		summed_images = 0
		for j in xrange(0, X.shape[0]):
			summed_images += r_nk[i, j]*X[j]
		div = 0
		for j in xrange(0, X.shape[0]):
			div += r_nk[i, j]
		if not div > 0:
			# print "Divisor is zero!"
			div = 1  # fix impossible arithmetics
		summed_images /= div
		mi_k[i] = summed_images

############################################################################

# the data matrix that contains all images
X = np.loadtxt("mixture.txt")
print X

# number of clusters
K = 3

# the mising values
pi_k = np.ones((K, 1)) / K

# choose the cluster centers (randomly)
mi_k = np.ones((K,2))
for i in xrange(0,K):
	mi_k[i] = X[np.random.random_integers(0,X.shape[0]-1)]

# covariances
sigma_k = np.ones((K,3,3))
for i in xrange(0,K):
	sigma_k[i] = np.identity(K)
print "sigma_k.shape:", sigma_k.shape


