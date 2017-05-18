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

from sklearn.preprocessing import PolynomialFeatures 
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


###############################################################################
# load the data
data = np.loadtxt("dataQuadReg2D_noisy.txt")
print "data.shape:", data.shape
np.savetxt("tmp.txt", data) # save data if you want to
# split into features and labels
X, y = data[:, :2], data[:, 2]
print "X.shape:", X.shape
print "y.shape:", y.shape

poly = PolynomialFeatures(2)
poly_X = poly.fit_transform(X)
print "poly_X.shape:", poly_X.shape

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
lambda_ = 10

error_arr = 5*[0]

mean_error = 7*[0]
for j in xrange(0,7):
	lambda_ = math.pow(10, j-3)
	for i in range(1,6):

		poly_X_sub = np.concatenate((poly_X[:10*(i-1)], poly_X[10*i:]))

		y_sub = np.concatenate((y[:10*(i-1)], y[10*i:]))

		identity = np.identity(poly_X_sub.shape[1])
		identity[0] = 0

		beta_ = mdot(inv(dot(poly_X_sub.T, poly_X_sub) + lambda_ * identity), poly_X_sub.T, y_sub)  # beta = (poly_X_sub^T*poly_X_sub)*poly_X_sub^T*y
		# print "Optimal beta:", beta_

		# prep for prediction
		X_grid = prepend_one(grid2d(-3, 3, num=30))
		# print "X_grid.shape:", X_grid.shape
		poly_X_grid = poly.fit_transform(X_grid[:,1:])
		# print "poly_X_grid.shape:", poly_X_grid.shape

		# Predict with trained model
		y_grid = mdot(poly_X_grid, beta_)
		# print "Y_grid.shape", y_grid.shape

		# Calculate mean squared error
		predicted_data_points = mdot(poly_X[:], beta_)
		error = np.linalg.norm(y - predicted_data_points)
		squared_error = error * error
		error_arr[i-1] = squared_error / predicted_data_points.shape[0]
		# print "error ", i, error_arr[i-1]
		print "lambda_", lambda_, "i", i, "error", error_arr[i-1]

	mean_error[j-1] = sum(error_arr)/5.
	print "mean_error", j, mean_error[j-1]
print "min mean_error", min(mean_error)
best_lambda = math.pow(10, mean_error.index(min(mean_error))-3)
print "best_lambda", best_lambda

# vis the result
print "mean_error", mean_error
fig = plt.figure()
ax = fig.add_subplot(111)
lambda_arr = 7*[0]
for i in xrange(0,7):
	lambda_arr[i] = math.pow(10, i-3)
ax.set_xscale("log")
ax.plot(lambda_arr, mean_error) # don’t use the 1 infront

# Fit model/compute optimal parameters beta
beta_ = mdot(inv(dot(poly_X.T, poly_X) + best_lambda * np.identity(poly_X.shape[1])), poly_X.T, y)  # beta = (poly_X^T*poly_X)*poly_X^T*y
print "Optimal beta:", beta_

# prep for prediction
X_grid = prepend_one(grid2d(-3, 3, num=30))
print "X_grid.shape:", X_grid.shape
poly_X_grid = poly.fit_transform(X_grid[:,1:])
print "poly_X_grid.shape:", poly_X_grid.shape

# Predict with trained model
y_grid = mdot(poly_X_grid, beta_)
print "Y_grid.shape", y_grid.shape

# Calculate mean squared error
predicted_data_points = mdot(poly_X[:], beta_)
error = np.linalg.norm(y - predicted_data_points)
squared_error = error * error
mean_squared_error = squared_error / predicted_data_points.shape[0]
print mean_squared_error

# vis the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # the projection part is important
ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid) # don’t use the 1 infront
ax.scatter(X[:, 1], X[:, 2], y, color="red") # also show the real data
# ax.scatter(X[:, 1], X[:, 2], predicted_data_points, color="green")
ax.set_title("predicted data")
plt.show()