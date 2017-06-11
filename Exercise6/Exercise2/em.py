import gmm
import numpy as np
import math
import matplotlib.pyplot as plt


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
			gamma_[x][k] = (theGMM.pi_[k] 
				* theGMM.getProbability(X, x, k)) / density_sum
	return gamma_


def MStep(theGMM, X, gamma_):

	sum_of_weights = [0 for k in xrange(theGMM.K)]
	for k in xrange(theGMM.K):
		for x in xrange(X.shape[0]):
			sum_of_weights[k] += gamma_[x][k]
	print sum(sum_of_weights)

	# 1. calculate the new weights
	for k in xrange(theGMM.K):
		theGMM.pi_[k] = sum_of_weights[k] /  X.shape[0]

	# 2. calculate the new means
	for k in xrange(theGMM.K):
		theGMM.mi_[k] = [0, 0]
		for x in xrange(X.shape[0]):
			theGMM.mi_[k] = np.add(theGMM.mi_[k], 
				mdot(1/sum_of_weights[k], gamma_[x][k], X[x]))

	# 3. calculate the new covariances
	for k in xrange(theGMM.K):
		theGMM.sigma_[k] = np.zeros((X.shape[1],X.shape[1]))
		for x in xrange(X.shape[0]):
			x_minus_mi = np.subtract(X[x], theGMM.mi_[k])
			theGMM.sigma_[k] = np.add(theGMM.sigma_[k], 
				mdot(1/sum_of_weights[k], gamma_[x][k], 
					mdot(x_minus_mi, x_minus_mi.T)))

def logLikelihood(X, theGMM):
	theLogLikelihood = 0
	for x in xrange(X.shape[0]):
		probabilty_sum = 0
		for k in xrange(theGMM.K):
			probabilty_sum += theGMM.pi_[k] * theGMM.getProbability(X, x, k)
		theLogLikelihood += math.log(probabilty_sum)
	return theLogLikelihood

def display_points(theGMM, X, gamma_):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	for x in xrange(X.shape[0]):
		clustercolor = 'b'
		for k in xrange(theGMM.K):
			if gamma_[x][k] is max(gamma_[x]):
				clustercolor = colors[k%len(colors)]
		ax.plot(X[x][0], X[x][1], '.'+clustercolor)
	for mean in theGMM.mi_:
		ax.plot(mean[0], mean[1], 'om')
	
	mean_x = [0, 0]
	for x in X:
		mean_x[0] += x[0]/len(X)
		mean_x[1] += x[1]/len(X)
	ax.plot(mean_x[0], mean_x[1], 'oc')

	plt.show()


