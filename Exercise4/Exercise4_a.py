#!/usr/bin/env python
# encoding: utf-8
"""
This is a mini demo of how to use numpy arrays and plot data.
NOTE: the operators + - * / are element wise operation. If you want
matrix multiplication use ‘‘dot‘‘ or ‘‘mdot‘‘!
"""
import numpy as np
from numpy import dot
from numpy.linalg import inv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  # 3D plotting


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
# load the data
data = np.loadtxt("data2Class.txt")
print "data.shape:", data.shape
np.savetxt("tmp.txt", data) # save data if you want to
# split into features and labels
X, y = data[:, :2], data[:, 2]
print "X.shape:", X.shape
print "y.shape:", y.shape

# 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # the projection arg is important!
ax.scatter(X[:, 0], X[:, 1], y, color="red")
ax.set_title("raw data")

plt.draw()

# show, use plt.show() for blocking

# prep for linear reg.
X = prepend_one(X)
print "X.shape:", X.shape

# Use lambda for regularization
lambda_ = 1000

### Fit model/compute optimal parameters beta

# initialize matrices and vectors
beta_ = np.zeros((X.shape[1], 1))	# 3x1
print "beta_.shape:", beta_.shape
p = np.ones((X.shape[0], 1))		# 200x1
print "p.shape:", p.shape
W = np.identity(X.shape[0])			# 200x200
print "W.shape:", W.shape
gradient = np.ones((X.shape[1], X.shape[1])) # 3x3
print "gradient.shape:", gradient.shape
hessian = np.ones((X.shape[0], X.shape[0]))  # 200x200
print "hessian.shape:", hessian.shape
# bring y into shaped form
y_shaped = np.copy(y)
y_shaped.shape = (y_shaped.shape[0], 1)
# print "y_shaped.shape:", y_shaped.shape
# print "p-y_shaped.shape:", (p-y_shaped).shape

# iterate Newton steps
for i in xrange(1,10):
	# calculate the probabilites p_i
	for j in xrange(0,X.shape[0]):
		p[j] = 1/(1+np.exp(-1*mdot(X[j,:].T, beta_)))
	# calculate the gradient
	gradient = mdot(X.T, p-y_shaped) + 2*lambda_*mdot(np.identity(X.shape[1]), beta_)
	# calculate matrix W
	for j in xrange(0,X.shape[0]):
		W[j, j] = p[j]*(1-p[j])
	# calculate the hessian
	hessian = mdot(X.T, W, X) + 2*lambda_*np.identity(X.shape[1])
	# print "hessian.shape:", hessian.shape
	# print "gradient.shape:", gradient.shape
	beta_ = beta_ - mdot(inv(hessian), gradient)
	print "Current beta:", beta_

print "Optimal beta:", beta_


# prep for prediction
X_grid = prepend_one(grid2d(-3, 3, num=30))
print "X_grid.shape:", X_grid.shape

# plot probability over 2D grid of test points
prob = np.ones((X_grid.shape[0], 1))
for j in xrange(0,X_grid.shape[0]):
		prob[j] = 1/(1+np.exp(-1*mdot(X_grid[j,:].T, beta_)))

# vis the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # the projection part is important
ax.scatter(X_grid[:, 1], X_grid[:, 2], prob, color="red") # also show the real data
ax.scatter(X[:, 1], X[:, 2], y, color="red") # also show the real data
ax.set_title("Probability function")
plt.show()
