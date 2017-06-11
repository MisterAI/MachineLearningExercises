import math
import numpy as np
import random


###############################################################################
# Helper functions
def mdot(*args):
	"""Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
	return reduce(np.dot, args)
############################################################################



class gaussianMixtureModel(object):
	"""
	A single gaussian mixture model having the parameters:
	
	K: number of clusters
	pi_: weights of the clusters (mixing values)
	mi_: means of the clusters
	sigma_: the covariances of the clusters
	"""
	def __init__(self, K, mi_, X):
		super(gaussianMixtureModel, self).__init__()

		self.K = K
		self.pi_ = [1./self.K for i in xrange(K)]
		self.mi_ = mi_
		self.sigma_ = [np.identity(X.shape[1]) for i in xrange(K)]

	def getProbability(self, X, i_x, j_k):
		x_minus_mi = np.subtract(X[i_x], self.mi_[j_k])
		
		# add some noise to the diagonal to avoid singularity
		self.sigma_[j_k] = np.add(self.sigma_[j_k], 
			mdot(0.01*random.random(), np.identity(X.shape[1])))
		sigma_inv = np.linalg.inv(self.sigma_[j_k])
		# print mdot(sigma_inv, self.sigma_[j_k])
		
		exponential = math.exp(-0.5 * mdot(
			x_minus_mi.T, sigma_inv, x_minus_mi))
		
		factor = 1/(math.pow(2*math.pi, X.shape[1]/2.) 
			* math.sqrt(np.linalg.det(self.sigma_[j_k])))
		
		probability = factor * exponential

		return probability
