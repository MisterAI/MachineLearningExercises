import gmm
import numpy as np
import math


###############################################################################
# Helper functions
def mdot(*args):
	"""Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
	return reduce(np.dot, args)
############################################################################


def EStep(theGMM, X):
	# the posterior probabilites
	gamma_ = [[0 for k in xrange(theGMM.K)] for x in xrange(X.shape[0])]

	for x in xrange(X.shape[0]):
		density_sum = 0
		for k in xrange(theGMM.K):
			density_sum += theGMM.pi_[k] * theGMM.getProbability(X, x, k)
		for k in xrange(theGMM.K):
			gamma_[x, k] = (theGMM.pi_[k] 
				* theGMM.getProbability(X, x, k)) / density_sum
	return gamma_


def MStep(theGMM, X, gamma_):

	sum_of_weights = [0 for k in xrange(theGMM.K)]
	for k in xrange(theGMM.K):
		for x in xrange(X.shape[0]):
			sum_of_weights[k] += gamma_[x, k]

	# 1. calculate the new weights
	for k in xrange(theGMM.K):
		theGMM.pi_[k] = sum_of_weights[k] /  X.shape[0]

	# 2. calculate the new mixture weights
	for k in xrange(theGMM.K):
		theGMM.mi_[k] = 0
		for x in X.shape[0]:
			theGMM.mi_[k] = np.add(theGMM.mi_[k], 
				mdot(1/sum_of_weights, gamma_[x,k], X[x]))

	# 3. calculate the new covariances
	for k in xrange(theGMM.K):
		theGMM.sigma_[k] = np.zeros((X.shape[0],X.shape[0]))
		for x in xrange(X.shape[0]):
			x_minus_mi = np.subtract(X[x], theGMM.mi_[k])
			theGMM.sigma_[k] = np.add(theGMM.sigma_[k], 
				mdot(1/sum_of_weights, gamma_[x,k], 
					mdot(x_minus_mi, x_minus_mi.T)))


def logLikelihood(X, theGMM):
	theLogLikelihood = 0
	for x in xrange(X.shape[0]):
		probabilty_sum = 0
		for k in xrange(theGMM.K):
			probabilty_sum += theGMM.pi_[k] * theGMM.getProbability(X, x, k)
		theLogLikelihood += math.log(probabilty_sum)
	return theLogLikelihood

