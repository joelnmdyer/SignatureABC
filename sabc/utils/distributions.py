import numpy as np
import torch
import torch.distributions as tds

class MA2UniformTriangle:

	def __init__(self):

		"""
		Inverted triangle prior with vertices at (-2,1), (2,1), (0,-1) for MA2
		model
		"""

		self.A = np.array([-2.,1.])
		self.B = np.array([0.,-1.])
		self.C = np.array([2.,1.])
		self.v = np.stack([self.A, self.B, self.C])
		assert self.v.shape == (3, 2)
		self.log_prob_val = np.log(1./4)

	def sample(self, n_samples=1):

		"""
		Generate samples uniformly from the prior

		n_samples is optional integer argument >= 1 determining the number of
		samples to be drawn from the density
		"""

		x = np.sort(np.random.rand(2, n_samples), axis=0)
		col = np.column_stack([x[0], x[1]-x[0], 1.-x[1]])
		p = col @ self.v
		return p

	def log_prob(self, th):

		"""
		Evaluate log density at th, which is (2,) or (n_samples, 2) np.array
		"""

		if len(th.shape) == 1:
			th1, th2 = th[0], th[1]
		elif len(th.shape) == 2:
			th1, th2 = th[:,0], th[:,1]
		else:
			raise RuntimeError("th must be either (2,) or (n_samples, 2)")
		mask = (
				  (th1 >= -2.) 
				* (th1 <=  2.)
				* (th2 >= -1 - th1)
				* (th2 >= th1 - 1)
				* (th2 <=  1) 
				* (th2 >= -1) # Redundant I think but who cares
			   )
		logprob = np.ones(th1.size)*self.log_prob_val
		logprob[~mask] = -float("inf")
		return logprob

class BoxUniform:

	def __init__(self, low, high):

		"""
		Implements a uniform hyper-rectangle in arbitrarily many dimensions

		low, high:		lists with boundaries for uniform distribution in each
						dimension
		"""

		self.__low = np.array(low).reshape(1, -1)
		self.__high = np.array(high).reshape(1, -1)
		self.__ndims = self.__low.size
		self.__range = self.__high - self.__low

	def sample(self, n_samples=1):

		"""
		Sample n_samples samples from the uniform hyper-rectangle

		n_samples is an optional integer argument >= determining the number of
		samples to be generated
		"""

		samples = np.random.random((n_samples, self.__ndims))*self.__range + self.__low
		return samples

	def log_prob(self, th):

		"""
		Evaluate the log density at th

		th is a numpy array of shape (d,) or (n_samples, d)
		"""

		mask = self.__low <= th <= self.__high
		if len(th.shape) == 1:
			length = 1
		elif len(th.shape) == 2:
			length = th.shape[0]
		else:
			raise RuntimeError("th must be either (d,) or (n_samples, d)")
		logprobs = np.empty(length)
		logprobs[mask] = 0.
		logprobs[~mask] = -float('inf')
		return logprobs 

class GSEBoxGamma:

	# Class for bivariate Gamma prior used for GSE model

	def __init__(self, lmbdas, nus):

		assert len(lmbdas) == 2, "This is not a general class, only for specific prior for GSE model"
		self.lmbda_bet, self.lmbda_gam = lmbdas
		self.nu_bet, self.nu_gam = nus
		self.beta_prior = tds.gamma.Gamma(self.lmbda_bet, self.nu_bet, validate_args=False)
		self.gamma_prior = tds.gamma.Gamma(self.lmbda_gam, self.nu_gam, validate_args=False)

	def sample(self, n_samples=1):

		"""
		Generates n_samples samples from the 2D Gamma distribution.

		n_samples is an integer >= 1 determining the number of samples to be
		generated
		"""

		if isinstance(n_samples, int):
			n_samples = (n_samples,)
		beta_sample = self.beta_prior.sample(n_samples)
		gamma_sample = self.gamma_prior.sample(n_samples)
		p = torch.stack((beta_sample, gamma_sample)).T
		if n_samples == (1,):
			p = p[0]
		p = p.numpy()
		return p

	def log_prob(self, th):

		"""
		Evaluate the log probability density function at th

		th must be a numpy array of shape (n_samples, 2) or (2,)
		"""

		if len(th.shape) == 2:
			th0, th1 = th[:,0], th[:,1]
			mask = (th0 > 0.) * (th1 > 0.)
		elif len(th.shape) == 1:
			th0, th1 = float(th[0]), float(th[1])
			mask = torch.tensor([th0 > 0., th1 > 0.])
		else:
			raise IndexError("This class is only for 2D Gamma prior for GSE model")
		th0, th1 = torch.as_tensor(th0), torch.as_tensor(th1)
		vals = (self.beta_prior.log_prob(th0) + self.gamma_prior.log_prob(th1)).reshape(-1)
		vals = vals.numpy()
		vals[~mask] = -float('inf')
		return vals

