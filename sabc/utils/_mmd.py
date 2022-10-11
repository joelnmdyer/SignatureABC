from numba import njit
import numpy as np


def _med_heuristic(x):
	"""
	x is the set of samples from which we want to compute the median heuristic
	"""
	z = np.empty(x.shape[0]**2)
	i = 0
	for xx in x:
		for xy in x:
			z[i] = np.sum((xx - xy)**2)/2
			i += 1
	return np.sqrt(np.median(z))

#@njit
def _get_rbf_bandwidth(x):
	"""
	x is the set of samples from which we want to compute the median heuristic
	"""
	z = np.empty(x.shape[0]**2)
	i = 0
	for xx in x:
		for xy in x:
			z[i] = np.sqrt(np.sum((xx - xy)**2))
			i += 1
	return np.median(z)

@njit
def _mv_gauss_rbf(xi, xj, c=1.):
	"""Gaussian radial basis function (rbf).
	Args:
		xi (float): i'th element of np.ndarray x.
		xj (float): j'th element of np.ndarray x.
		c (float, optional): bandwidth parameter for RBF kernel.
	Returns:
		float: evaluation of rbf.
	"""
	diff = xi-xj
	dot_diff = diff.dot(diff)
	return np.exp(-dot_diff/(2*c**2))

@njit
def compute_mmd(x, y, c):
	"""Produces an unbiased estimate of the maximum mean discprepancy (mmd) 
	between two distributions using samples from each. See p.4 [1].
	[1] K2-ABC: Approximate Bayesian Computation with Kernel Embeddings.
		Park, M., Jitkrittum, W., Sejdinovic, D. 2015
	Args:
		x (np.ndarray): (n_samples, n_dims) samples from first distribution.
		y (np.ndarray): (n_samples, n_dims) samples from second distribution.
		c (float):		value of bandwidth parameter in rbf kernel.
	Returns:
		float: The mmd estimate.
	"""
	n_x = x.shape[0]
	n_y = y.shape[0]

	factor1 = 0
	for i in range(n_x):
		for j in range(n_x):
			if (j == i): continue
			factor1 += _mv_gauss_rbf(x[i], x[j], c)
	factor1 /= (n_x*(n_x-1))

	factor2 = 0
	for i in range(n_y):
		for j in range(n_y):
			if (j == i): continue
			factor2 += _mv_gauss_rbf(y[i], y[j], c)
	factor2 /= (n_y*(n_y-1))

	factor3 = 0
	for i in range(n_x):
		for j in range(n_y):
			factor3 += _mv_gauss_rbf(x[i], y[j], c)
	factor3 *= 2/(n_x*n_y)

	return factor1 + factor2 - factor3

