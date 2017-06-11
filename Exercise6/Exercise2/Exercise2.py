import numpy as np
import math
import gmm
import em


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

convergence = False

while not convergence:
	old_logLikelihood = 0
	gamma_ = em.EStep(theGMM, X)
	em.MStep(theGMM, X, gamma_)
	new_logLikelihood = em.logLikelihood(X, theGMM)
	
	# check for convergence
	convergence = abs(new_logLikelihood - old_logLikelihood) < 0.01  #tbc

