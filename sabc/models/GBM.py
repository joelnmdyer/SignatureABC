import functools as ft
import numpy as np
import os
import scipy.stats
import torch
from tqdm import trange

basedir = os.path.dirname(os.path.abspath(__file__))

# Summary stats
def incs(x):
	return x#np.diff(x)

def mean_incs(x):
	return np.mean(incs(x))

def var_incs(x):
	return np.var(incs(x))

def autocovariances_to_lag5_incs(x):
    return sttsa.acovf(incs(x), fft=False, nlag=5)

_statistics = [mean_incs, var_incs, autocovariances_to_lag5_incs]
statistics = [lambda x: np.power(x, j) for j in range(1, 5)]

# Prior distributions
lbs = [-1., 0.2]
ubs = [1., 2.]
n_pars = len(lbs)
mu_default = 0.2
sigma_default = 0.5
S0_default = 10

class Model:

	def __init__(self):
		self.mu = torch.as_tensor(mu_default)
		self.sigma = torch.as_tensor(sigma_default)
		# Always use same starting point
		self.S0 = torch.as_tensor(S0_default)

	def simulate(self, pars=None, T=100, seed=None):
		"""
		Generates a (batch of) samples of univariate geometric Brownian motion (GBM).

		The shapes of S0, mu, sigma don't need to match, but they must be broadcastable with each other.

		Arguments:
			S0: The initial value of the GBM. Must be a Python float or a tensor.
			mu: The drift coefficient. Must be a Python float or a tensor.
			sigma: The volatility coefficient. Must be a Python float or a tensor.
			T: The terminal time of the GBM. Must be a Python float or a scalar tensor.
			N: The number of samples of the GBM. Must be a Python float or a scalar tensor.

		Returns:
			A tensor of shape (*, N). Taking '*' to be some k dimensions indexed by i_1, ..., i_k, then the i1, ..., i_k
			element of the returned tensor will be a univariate GBM sampled at N uniformly spaced times between 0 and T,
			drift coefficient mu[i_1, ..., i_k], volatility coefficient sigma[i_1, ..., i_k], and initial value
			S0[i_1, ..., i_k].

		Example:
			S0 = torch.rand(10)
			out = gbm(S0, mu=3, sigma=5, T=10, N=100)
		"""

		if not (seed is None):
			torch.manual_seed(seed)
			np.random.seed(seed)

		# Convert all arguments to tensors
		if not (pars is None):
			self.mu, self.sigma = float(pars[0]), float(pars[1])
			self.mu = torch.as_tensor(self.mu)
			self.sigma = torch.as_tensor(self.sigma)
			self.S0, self.mu, self.sigma = torch.broadcast_tensors(self.S0, 
																   self.mu,
																   self.sigma)

		max_t = torch.as_tensor(1.)
		T = torch.as_tensor(T)

		# Check sizes and values of them
		assert max_t.shape == T.shape == (), "max_t and T must be scalars."
		assert T >= 2, "T must be at least 2."
		assert not T.is_floating_point(), "T must not be floating point."

		# Convert all arguments to the same dtype, and at least a float.
		dtype = ft.reduce(torch.promote_types,
						  [self.S0.dtype,
						   self.mu.dtype,
						   self.sigma.dtype,
						   max_t.dtype,
						   torch.float
						  ])
		self.S0 = self.S0.to(dtype)
		self.mu = self.mu.to(dtype)
		self.sigma = self.sigma.to(dtype)
		max_t = max_t.to(dtype)

		# Actually do the simulation
		dt = max_t / (T - 1)
		S0 = self.S0.unsqueeze(-1)
		mu = self.mu.unsqueeze(-1)
		sigma = self.sigma.unsqueeze(-1)
		term = mu - 0.5*sigma**2
		means = term.repeat_interleave(T - 1, dim=-1) * dt
		stds = sigma.repeat_interleave(T - 1, dim=-1) * dt.sqrt()
		X = torch.normal(means, stds)
		X = X.cumsum(dim=-1)
		S = torch.cat([S0, S0 * X.exp()], dim=-1).numpy().reshape(-1,1)
		ts = np.arange(S.size).reshape(-1,1)
		ts = ts / np.max(ts)
		return np.concatenate((S, ts), axis=-1)

def loglike(y, th):

	mu, sig = th
	dt = 1. / (len(y) - 1)
	ll = 0
	for j in range(1, len(y)):
		const = 2*(sig**2)*dt
		ll += -np.log(y[j]*np.sqrt(np.pi*const))
		ll += -(
				( np.log(y[j]) - np.log(y[j-1]) - (mu*dt - const/4.) )**2 
			   ) / const
	return ll

def sample_from_post(y, n_samples=10_000, x0=None, cov=np.eye(2),
					 seed=1):

	"""
	For MCMC sampling from posterior
	"""

	np.random.seed(seed)

	if x0 is None:
		x0 = np.array([mu_default, sigma_default])

	# Gaussian innovations
	proposal = scipy.stats.multivariate_normal

	xs = np.zeros((x0.size, n_samples))
	xs[:, 0] = x0

	x_ = x0
	rev_logpost = loglike(y,x_) + prior.log_prob(torch.tensor(x_).float())

	test_output = 0.
	acceptance_rate = 0.
	neg_inf = float("-inf")

	t = trange(1, n_samples, position=0, leave=True)
	for n in t:
		# Propose new point
		x = proposal.rvs(mean=x_, cov=cov)
		new_logpost = loglike(y,x) + prior.log_prob(torch.tensor(x).float())
		# Reject if outside prior range
		if new_logpost == neg_inf:
			test_output += 1
			xs[:, n] = x_
			continue
		# Find log-pdf of new point from proposal
		new_logpdf = proposal.logpdf(x, mean=x_, cov=cov)
		# Find log-pdf of old point given new point
		rev_logpdf = proposal.logpdf(x_, mean=x, cov=cov)
		# Acceptance probability
		log_alpha = new_logpost + rev_logpdf - rev_logpost - new_logpdf
		if np.random.rand() >= np.exp(log_alpha):
			# Fail, reject proposal
			xs[:, n] = x_
			continue
		# Success
		xs[:, n] = x
		x_ = x 
		rev_logpost = new_logpost
		acceptance_rate += 1
		t.set_postfix({"Acc.:": acceptance_rate/n,
					   "test: ": test_output/n,
					   "pos:":x_})
		t.refresh()  # to show immediately the update

	return xs
