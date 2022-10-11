from collections import namedtuple
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
from tqdm import trange

RParam = namedtuple("RParam", ["beta", "gamma", "max_time", "n", "i0"])
default_param = RParam(beta=1e-2, gamma=1e-1, max_time=50, n=10**2, i0=1)

basedir = os.path.dirname(os.path.abspath(__file__))
lbs=[0,0]
ubs=[None,None]
lambdas, nus = [0.1, 0.2], [2, 0.5]
n_pars = len(lbs)

@njit
def _accum_integrals(ns, ni, dt, first):

	"""
	This builds up the two integrals we're interested in for posterior
	inference over the course of the simulation.
	"""

	# Only count from first infection
	if first:
		return 0., 0.
	else:
		# They are piecewise constant over interval
		# Doing this for many time steps and large population gives large
		# values though
		return ns*ni*dt, ni*dt	

@njit
def _step(model, ns, ni, nr, t, seed):

	if t == 0:	
		if not (seed is None):
			np.random.seed(seed)

	p_infection = model.beta * ns * ni
	p_recovery = model.gamma * ni
	p_total = p_infection + p_recovery

	dt = np.random.exponential(1 / p_total)
	t += dt

	if np.random.rand() < p_infection / p_total:
		ns -= 1
		ni += 1

	else:
		ni -= 1
		nr += 1
	return ns, ni, nr, t

#@njit
def _simulate(model, nss, nis, nrs, post, xy_int, y_int, niend, nrend, seed):

	t = 0
	ts = [0]

	ns, ni, nr = nss[0], nis[0], nrs[0]
	while (t < model.max_time) and (ni > 0):

		# Update state
		new_ns, new_ni, new_nr, new_t = _step(model, ns, ni, nr, t, seed)
		if new_t > model.max_time:
			ts.append(model.max_time)	
			nss.append(ns)
			nis.append(ni)
			nrs.append(nr)
			break
		ts.append(new_t)

		# Accumulate integrals for posterior
		if post:
			# We want the values of ns and ni since the last event, and
			# the time since last event
			# For fairness, we make sure to use smallest of max_time and new_t
			# since different simulations will have a different "last event
			# time"
			xy, y = _accum_integrals(ns, ni, new_t-t, t==0)
			xy_int += xy
			y_int  += y

		ns, ni, nr = new_ns, new_ni, new_nr
		# Store variables
		nss.append(ns)
		nis.append(ni)
		nrs.append(nr)

		# Update state if you didn't do it above
		t = new_t

	if (t < model.max_time) and (ni == 0):
		# If we get to here, then the simulation didn't end but there are no
		# more infected individuals. In that case, you need to observe the
		# same state for the remaining time steps
		ts.append(model.max_time)
		nss.append(ns)
		nis.append(ni)
		nrs.append(nr)
		# We also don't need to worry about updating the integrals, because
		# the integrands are zero if ni == 0

	# Want to return integrals and TOTAL number infected and recovered
	# throughout pandemic. R is absorbing so nr is fine for latter
	return xy_int, y_int, ts

class Model:

	"""
	The GSE model simulates an epidemics in a population of n individuals. We
	use the Gillespie algorithm to generate a continuous time Markov chain
	process. Time steps at events happen are generated from an exponential
	distribution. At each sampled time either a new infection or recovery
	occurs, with probabilities p_infection and p_recovery. References: 
	https://eprints.lancs.ac.uk/id/eprint/26392/1/kypraios.pdf page 44-46

	Input:

	- dummy: 	does nothing, only here for consistency with other model
				interfaces and so code elsewhere doesn't break
	- pars:		RParam instance containing parameter values
	- post:		bool, indicates whether we want to track values relevant to
				posterior inference, i.e. the integrals

	"""

	def __init__(self, dummy=None, pars=default_param, post=False):
		self.pars = pars
		# Boolean flag to indicate whether we want to evaluate integrals on
		# the fly for posterior inference
		self.post = post
		self.reset()

	def reset(self):
		self.ni = self.pars.i0
		self.ns = self.pars.n - self.pars.i0
		self.nr = 0
		self.xy_int = 0.
		self.y_int = 0.

	def simulate(self, pars=None, T=None, seed=None):

		"""
		This function runs the epidemic simulation

		Input:

			pars: epidemic parameters, pars[0] = beta, pars[1] = gamma
			T (int): ignored
			seed (int): for initialising RNG

		Outputs:
		
			sir: tensor of shape (1, -1, 3), -1 indicating unknown length
		"""

		if not (pars is None):
			self.pars = RParam(beta=float(pars[0]), gamma=float(pars[1]),
							   n=default_param.n, i0=default_param.i0,
							   max_time=default_param.max_time)

		self.reset()

		self.nss, self.nis, self.nrs = [], [], []
		self.nss.append(self.ns)
		self.nis.append(self.ni)
		self.nrs.append(self.nr)
		self.xy_int, self.y_int, ts = _simulate(self.pars, self.nss, self.nis,
											self.nrs, self.post, self.xy_int,
											self.y_int, self.ni, self.nr,
											seed)

		if self.post:
			self.ni, self.nr = self.nss[0] - self.nss[-1], self.nrs[-1]

		return np.stack((self.nis, self.nrs, ts)).T.astype(float)

	def log_post(self, betagamma, lambdas=lambdas, nus=nus):

		"""
		Evaluate the posterior at parameters betagamma = [beta, gamma],
		assuming Gamma priors over both beta and gamma, each with parameters
		lambdas = [lmbda_bet, lmbda_gam] and nus = [nu_bet, nu_gam]. See page
		58 of https://eprints.lancs.ac.uk/id/eprint/26392/1/kypraios.pdf.
		"""

		beta, gamma = betagamma
		if (beta <= 0.) or (gamma <= 0.):
			return float("-inf")
		lmbda_bet, lmbda_gam = lambdas
		nu_bet, nu_gam = nus
		# Evaluate terms of posterior
		term1 = np.log(beta)*(lmbda_bet+self.ni-2)
		term2 = -beta*(self.xy_int + nu_bet)
		term3 = np.log(gamma)*(lmbda_gam+self.nr-1)
		term4 = -gamma*(self.y_int + nu_gam)
		# Posterior proportional to below
		return term1 + term2 + term3 + term4

	def samples_from_post(self, n_samples=10_000, seed=1):
	
		"""
		Samples from marginal posteriors
		"""

		torch.manual_seed(seed)

		lmbda, nu = lambdas[0], nus[0]
		term1 = self.ni - 1
		term2 = self.xy_int
		beta_dist = torch.distributions.gamma.Gamma(lmbda+term1, nu+term2)

		lmbda, nu = lambdas[1], nus[1]
		term1 = self.nr
		term2 = self.y_int
		gamma_dist = torch.distributions.gamma.Gamma(lmbda+term1, nu+term2)

		betas = beta_dist.sample((n_samples,)).reshape(1,-1)
		gammas = gamma_dist.sample((n_samples,)).reshape(1,-1)
		return torch.cat((betas, gammas))

	def plot_posteriors(self, n_samples=10_000, seed=1):

		"""
		Attempt at marginal plotting
		"""

		samples = self.samples_from_post(n_samples, seed=seed)
		plot_samples = pd.DataFrame(samples.numpy().T)
		names = [r"$\beta$", r"$\gamma$"]
		plot_samples.columns = names
		fig, axes = plt.subplots(1, 2, figsize=(15,3.5))
		for i in range(2):
			sns.kdeplot(x=names[i], data=plot_samples, ax=axes[i], fill=True,
						clip=[0,None])
		return samples

