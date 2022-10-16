from numba import njit
import numpy as np
import scipy.stats
from tqdm import trange

@njit
def _simulate(alpha, beta, y, seed):

	"""
	alpha - 	in [0,1]
	beta  -		in [0,1]
	y	  -		shape (T, N, N)
	seed  -		integer, random seed
	"""

	if not (seed is None):
		np.random.seed(seed)

	N = y.shape[1]

	# Maybe skip this block
	for j in range(N):
		i = 0
		while i < j:
			prob = alpha / (alpha + beta)
			outcome = int(np.random.random() < prob)
			y[1, i, j] = outcome
			y[1, j, i] = outcome	
			i += 1

	# Simulate dynamics
	for t in range(2, y.shape[0]):
		for j in range(N):
			i = 0
			while i < j:
				if y[t - 1, i, j] == 0:
					prob = alpha
				else:
					prob = 1 - beta
				outcome = int(np.random.random() < prob)
				y[t, i, j] = outcome
				y[t, j, i] = outcome	
				i += 1
	return y

@njit
def _compute_ll(alpha, beta, y):

	lalpha, lbeta, lalta = np.log(alpha), np.log(beta), np.log(alpha + beta)
	loneal, lonebe = np.log(1. - alpha), np.log(1. - beta)

	N = y.shape[1]
	# The next two lines should account for the cumsum and normalise
	y = y * (y.shape[0] + 1)
	y = np.diff(y, axis=0)

	ll = 0.

	for j in range(N):
		i = 0
		while i < j:
			if y[0, i, j] == 1:
				ll += lalpha - lalta
			else:
				ll += lbeta - lalta
			i += 1

	for t in range(1, y.shape[0]):
		for j in range(N):
			i = 0
			while i < j:
				if y[t - 1, i, j] == 1:
					if y[t, i, j] == 1:
						ll += lonebe
					else:
						ll += lbeta
				else:
					if y[t, i, j] == 1:
						ll += lalpha
					else:
						ll += loneal
				i += 1	
	return ll


class Model:

	def __init__(self, N=20):

		self._N = N

	def simulate(self, pars=None, T=None, seed=None):

		if not (pars is None):
			alpha, beta = [float(p) for p in pars]
			assert (0. < alpha) and (alpha <= 1.) and (0. < beta) and (beta <= 1.)
		else:
			alpha, beta = 0.4, 0.7
		if not (T is None):
			assert isinstance(T, int) and (T > 0), "T must be positive int"
		else:
			T = 100
		y = np.zeros((T+2, self._N, self._N), dtype=np.int)
		y = _simulate(alpha, beta, y, seed)
		y = np.cumsum(y, axis=0)/(T+1)
		return y
