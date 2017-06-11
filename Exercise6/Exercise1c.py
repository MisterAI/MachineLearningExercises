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

###############################################################################

def display_means(mi_k, image_height, K):
	w,h = divideSquare(K)
	fig = pylab.figure()
	for i in xrange(0,K):
		fig.add_subplot(w,h,i+1)
		pylab.imshow(np.stack(np.split(mi_k[i], image_height)), cmap='gray')
	pylab.show()

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

###############################################################################

# the data matrix that contains all images
X = np.ones((136,38880))

# read all images into data matrix
image_path = "./yalefaces_cropBackground/"
i = 0
for file_name in os.listdir(image_path):
	curr_image = plt.imread(image_path + file_name)
	X[i] = np.concatenate(curr_image).T[0]
	# print X[i]
	i += 1
print "Done loading images."




# calculate the mean faces
mean_faces = np.zeros((X.shape[1],1))
# print "mean_faces.shape:", mean_faces.shape
i = 0
for column in X.T:
	mean_faces[i] = np.mean(column) / X.shape[1]
	i += 1
print "Done computing mean faces."
# print "mean_faces.shape:", mean_faces.shape
# print "X.shape:", X.shape

# calculate the centered matrix
X_centered = X - mdot(np.ones((X.shape[0],1)), mean_faces.T)
print "Done computing centered X."

# calculate the singular value decomposition
# for num_eigenvalues in [5,10,25,50,100,165]:
for num_eigenvalues in [20]:
	# num_eigenvalues = 100
	u, s, vt = sla.svds(X_centered, k=num_eigenvalues)
	print "vt.shape", vt.shape
	print "Done computing singular value decomposition."

	# find num_eigenvalues-dimensional representation
	Z = mdot(X_centered, vt.T)

	##########
	# do k-means clustering on principal components
	##########
	error_values = []
	for K in xrange(4,16):
		
		# Initialize the K cluster centers randomly
		mi_k = 255 * np.random.rand(K, vt.shape[1])

		# Display the random mean faces
		image_height = 243
		# display_means(mi_k, image_height, K)

		# the indicator variables that assign the images a cluster
		r_nk = np.zeros((K, vt.shape[0]))
		
		# Assign the images to the clusters
		assign_to_clusters(vt, mi_k, r_nk)

		# Fix the means
		adapt_means(mi_k, r_nk, vt, K)

		# iterate nine more times (until convergence)
		iteration_steps = 9
		for q in xrange(1,iteration_steps + 1):
			# Display the random mean faces
			# display_means(mi_k, image_height, K)

			# Reassign the images to the clusters
			error = assign_to_clusters(vt, mi_k, r_nk)
			if iteration_steps is q:
				print "Overall error for ", K, "clusters:", error
				error_values.append(error)

			# Fix the means
			adapt_means(mi_k, r_nk, vt, K)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	K_values = xrange(4,16)
	ax.plot(K_values, error_values)
	pylab.show()
