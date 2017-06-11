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

# the data matrix that contains all images
X = np.ones((166,77760))

original_images = []

# read all images into data matrix
image_path = "./yalefaces/"
i = 0
for file_name in os.listdir(image_path):
	if not file_name.endswith(".txt"):
		original_images.append(plt.imread(image_path + file_name))
		curr_image = original_images[len(original_images) - 1]
		X[i] = np.concatenate(curr_image)
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
print "mean_faces.shape:", mean_faces.shape
# print "X.shape:", X.shape

# calculate the centered matrix
X_centered = X - mdot(np.ones((X.shape[0],1)), mean_faces.T)
print "Done computing centered X."

# calculate the singular value decomposition
# for num_eigenvalues in [5,10,25,50,100,165]:
for num_eigenvalues in [60]:
	# num_eigenvalues = 100
	u, s, vt = sla.svds(X_centered, k=num_eigenvalues)
	print "Done computing singular value decomposition."

	# find num_eigenvalues-dimensional representation
	Z = mdot(X_centered, vt.T)

	# reconstruct the faces
	X_reconstructed = mdot(np.ones((X.shape[0],1)), mean_faces.T) \
						+ mdot(Z, vt)
	print "Done reconstructing images."


	# display the reconstructed images
	image_height = 243
	reconstructed_images = []
	i = 0
	for big_row in X_reconstructed:
		reconstructed_images.append(
			np.stack(np.split(big_row, image_height)))

	# compute reconstruction error
	reconstruction_error = 0
	for i in xrange(0, X.shape[0]):
		reconstruction_error += np.linalg.norm(original_images[i] - reconstructed_images[i])**2
	print "Reconstruction error /10^6; #eigenvalues:",\
		num_eigenvalues,\
		"reconstruction error:",\
		reconstruction_error / 10**6

for i in xrange(0,len(original_images)):
	fig = pylab.figure()
	fig.add_subplot(1,2,1)
	pylab.imshow(original_images[i], cmap='gray')
	fig.add_subplot(1,2,2)
	pylab.imshow(reconstructed_images[i], cmap='gray')
	pylab.show()

