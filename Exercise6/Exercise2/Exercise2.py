import numpy as np
import math
import gmm
import em
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def plot_gamma(gamma_):
	# 3D plotting
	fig = plt.figure()
	ax = Axes3D(fig)
	for x in xrange(len(gamma_)):
		for k in xrange(len(gamma_[0])):
			ax.scatter(x, k, gamma_[x][k], color="red")
	ax.set_title("gamma values")
	plt.show()


# the data matrix that contains all images
X = np.loadtxt("mixture.txt")
print "X.shape", X.shape

# number of clusters
K = 3

# choose the cluster centers (randomly)
mi_ = np.ones((K,2))
for i in xrange(0,K):
	mi_[i] = X[np.random.random_integers(0,X.shape[0]-1)]

theGMM = gmm.gaussianMixtureModel(K, mi_, X)
# print theGMM.mi_
# print theGMM.pi_
# print theGMM.sigma_


convergence = False

i = 0
old_logLikelihood = 0
while not convergence:
	print i+1, "th iteration"
	# if i is 0:
	# 	gamma_ = gamma_ = [[0 for k in xrange(theGMM.K)] for x in xrange(X.shape[0])]
	# 	for x in xrange(X.shape[0]):
	# 		gamma_[x][np.random.randint(K)] = 1.
	# else:
	gamma_ = em.EStep(theGMM, X)
	# plot_gamma(gamma_)
	em.MStep(theGMM, X, gamma_)
	em.display_points(theGMM, X, gamma_)

	# print theGMM.mi_
	# print theGMM.pi_
	# print theGMM.sigma_

	new_logLikelihood = em.logLikelihood(X, theGMM)
	
	# check for convergence
	print "error: ", abs(new_logLikelihood - old_logLikelihood)
	convergence = abs(new_logLikelihood - old_logLikelihood) < 0.01  #tbc
	old_logLikelihood = new_logLikelihood
	i+=1
