from numba import njit
import numpy as np
from scipy import stats

@njit
def _get_utility(g, b, x, R, i):
	return (x[i + 2] - R * x[i + 1]) * (g * x[i] + b - R * x[i + 1])

@njit
def _get_mean(n_h, g, b, x, R, i):
	return ((n_h * (g * x[i + 2] + b)).sum()) / R 

@njit
def _get_numbers(U_h, beta):

	exponents = beta * U_h
	exponents -= exponents.max()
	n_h = np.exp(exponents)
	return n_h / n_h.sum()

@njit
def brock_hommes(g_1, b_1, g_2, b_2, g_3, b_3, g_4, b_4, r=0.01, beta=120.,
				 sigma=0.04, p_star=0., T=100, seed=None):
	
	# Set seed for RNG
	if not seed is None:
		np.random.seed(seed)
	
	R = 1 + r
	
	# Model parameters
	g = np.array([g_1, g_2, g_3, g_4])
	b = np.array([b_1, b_2, b_3, b_4])
	
	# This is the (deviation from fundamental price of the) asset price
	x = np.zeros(T)
		
	# Random shocks
	epsilon = np.random.randn(T) * sigma / R
 
	# Simulate
	for i in range(T - 3):
		U_h = _get_utility(g, b, x, R, i)
		n_h = _get_numbers(U_h, beta)
		x[i + 3] = epsilon[i + 3] + _get_mean(n_h, g, b, x, R, i)
	
	# Actual prices
	return x + p_star


class Model:
	def __init__(self, beta=120., sigma=0.04, r=0.01):
		self.beta = beta
		self.sigma = sigma
		self.r = r
		self.R = 1. + self.r
		self.g1 = 0.
		self.b1 = 0.
		self.g4 = 1.01
		self.b4 = 0.

	def log_likelihood(self, y, pars):
		assert len(y.shape) == 2, "Reshape so that y of shape (T, 1)"
		# Parameter values
		g2, b2, g3, b3 = [float(pars[i]) for i in range(4)]
		# Always observe 0s in the first three time steps
		y_ = np.zeros((3, 1))
		scale = self.sigma / self.R
		b = np.array([self.b1, b2, b3, self.b4])
		g = np.array([self.g1, g2, g3, self.g4])
		ll = 0.
		for t in range(y.shape[0]):
			U_h = _get_utility(g, b, y_, self.R, 0, 0)
			n_h = _get_numbers(U_h, self.beta)
			mean = _get_mean(n_h, g, b, y_, self.R, 0, 0)
			ll += stats.norm.logpdf(y[t], loc=mean, scale=scale)
			y_[:2] = y_[-2:]
			y_[-1] = y[t]
		return ll

	def simulate(self, pars=None, T=100, seed=None):
		g2, b2, g3, b3 = [float(pars[i]) for i in range(4)]

		x = brock_hommes(self.g1, self.b1, g2, b2, g3, b3, self.g4, self.b4, self.r,
						 self.beta, sigma=self.sigma, T=T, seed=seed)
		x = np.expand_dims(x[2:], axis=-1)
		ts = np.arange(x.size).reshape(-1,1)
		ts = ts / np.max(ts)
		x = np.concatenate((x.reshape(-1,1), ts), axis=-1)
		return x
