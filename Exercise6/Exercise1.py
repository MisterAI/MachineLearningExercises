import matplotlib.pyplot as plt
import scipy as sp
import os
import numpy as np
import scipy.sparse.linalg as sla;
import pylab

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


###############################################################################

def display_means(mi_k, image_height):
	for image in mi_k:
		fig = pylab.figure()
		pylab.imshow(np.stack(np.split(image, image_height)), cmap='gray')
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
	print "Overall error:", overall_error

def adapt_means(mi_k,r_nk,X, K):
	for i in xrange(0, K):
	summed_images = 0
	for j in xrange(0, X.shape[0]):
		summed_images += r_nk[i, j]*X[j]
	div = 0
	for j in xrange(0, X.shape[0]):
		div += r_nk[i, j]
	print "div", div
	summed_images /= div
	mi_k[i] = summed_images


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

# Initialize the K cluster centers randomly
K = 4
mi_k = 255 * np.random.rand(K, X.shape[1])

# Display the random mean faces
image_height = 243
for image in mi_k:
	fig = pylab.figure()
	pylab.imshow(np.stack(np.split(image, image_height)), cmap='gray')
	# pylab.show()
	

# the indicator variables that assign the images a cluster
r_nk = np.zeros((K, X.shape[0]))

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
print "Overall error:", overall_error
print "r_nk", r_nk

# Fix the means
for i in xrange(0, K):
	summed_images = 0
	for j in xrange(0, X.shape[0]):
		summed_images += r_nk[i, j]*X[j]
	div = 0
	for j in xrange(0, X.shape[0]):
		div += r_nk[i, j]
	print "div", div
	summed_images /= div
	mi_k[i] = summed_images

# iterate nine more times (until convergence)
iteration_steps = 9
for q in xrange(1,iteration_steps):
	# Display the random mean faces
	for image in mi_k:
		fig = pylab.figure()
		pylab.imshow(np.stack(np.split(image, image_height)), cmap='gray')
		# pylab.show()

	# Reassign the images to the clusters
	overall_error = 0
	i = 0
	j = 0
	min_k = 0
	for image in X:
		r_min = float("inf")
		min_k = 0
		j = 0
		for center in mi_k:
			squared_diff = mdot((image - center).T, image - center)
			# find the center with the minimal distance
			if r_min > squared_diff:
				print r_min
				r_min = squared_diff
				min_k = j
			j += 1
		overall_error += r_min
		mi_k[min_k, i] = 1
		i += 1

	# Fix the means
	for i in xrange(0, K):
		summed_images = 0
		for j in xrange(0, X.shape[0]):
			summed_images += r_nk[i, j]*X[j]
		div = 0
		for j in xrange(0, X.shape[0]):
			div += r_nk[i, j]
		summed_images /= div
		mi_k[i] = summed_images

