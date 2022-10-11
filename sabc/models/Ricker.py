from collections import namedtuple
import math
from numba import njit
import numpy as np
import os
from sklearn import linear_model
from scipy import stats
from scipy.special import logsumexp
import statsmodels.tsa.stattools as sttsa
import torch

from tqdm import trange

basedir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/Ricker")
data = np.loadtxt(os.path.join(basedir, "obs.txt"))
# Remove basepoint
_data = data[1:]
# ===================
# HAND SUMMARIES
# ===================
diff_data = np.diff(_data)
diff_data = np.sort(diff_data)
_X = np.concatenate((diff_data.reshape(-1,1), np.power(diff_data, 2).reshape(-1,1),
					np.power(diff_data, 3).reshape(-1,1)), axis=1)

def num_greater_than_10(x):
    return np.sum(x > 10)

median = lambda x: np.median(x)

npmax = lambda x: np.max(x)

def quantile_75(x):
    return np.quantile(x, 0.75)

def mean_greater_than_1(x):
    return np.mean(x > 1)

def mean(x):
	return np.mean(np.array(x), axis=-1)

def number_of_zeros(x):
    return np.sum(np.array(x)==0., axis=-1)

def autocovariances_to_lag5(x):
    return sttsa.acovf(x, fft=False, nlag=5)

def regression_parameters(x):
    y = np.power(x[1:], 0.3)
    X = np.concatenate((np.power(x[:-1], 0.3).reshape(-1,1),
                        np.power(x[:-1], 0.6).reshape(-1,1)), axis=1)
    return linear_model.LinearRegression().fit(X, y).coef_

def cubic_regression_parameters(x):
	diff_x = np.diff(x)
	y = np.sort(diff_x)
	# _X is computed from the observed data, see above
	return linear_model.LinearRegression().fit(_X, y).coef_
	
statistics = [autocovariances_to_lag5, cubic_regression_parameters, regression_parameters, mean, number_of_zeros]

# =================
# PRIOR
# =================
lbs = [3.,0.,0.]
ubs = [8.,20.,0.6]

n_pars = len(lbs)

RParam = namedtuple("RParam", ["lr", "phi", "sig", "N0", "e0"])
default_param = RParam(
	lr  = 4.,
	phi = 10.,
	sig = 0.3,
	N0  = 1.0,
	e0 = 0
)

@njit
def r_forward(pars, N, e):
	return np.exp(pars.lr + np.log(N) - N + e)

@njit
def r_measure(pars, N):
	return np.random.poisson(lam=pars.phi*N)

#@njit
def log_d_measure(pars, N, y):
	return stats.poisson._logpmf(k=y, mu=pars.phi*N)

@njit
def _simulate(pars, y, N, seed=None):	
	if seed is not None:
		np.random.seed(seed)

	T = y.size - 1
	N[0] = pars.N0
	es = np.zeros(T+1)
	es[0] = pars.e0
	es[1:] = np.random.normal(0, scale=pars.sig, size=T)
	for t in range(1, T+1):
		N[t] = r_forward(pars, N[t-1], es[t-1])
		y[t] = r_measure(pars, N[t])
			  
	return y, N[1:]


class Model:
	def __init__(self, pars=default_param):
		self.pars = pars

	def simulate(self, pars=None, T=50, seed=None):
		if pars is not None:
			self.pars = RParam(lr=float(pars[0]), phi=float(pars[1]),
											sig=float(pars[2]), 
											N0  = 1.0,
											e0 = 0)

		y = np.zeros(T+1)
		N = np.zeros(T+1)
		y, N = _simulate(self.pars, y, N, seed=seed)
		ts = np.arange(y.size).reshape(-1,1)
		ts = ts / float(np.max(ts))
		y = np.concatenate((y.reshape(-1,1), ts), axis=-1)
		# Amend so it returns state of RNG and value of N? Then simulations
		# can be continued if so desired
		return y#, N

class BootstrapPF():

	def __init__(self, r_forward, r_measure, log_d_measure, model):
		self.forward = r_forward
		self.emission = r_measure
		self.logmeasure = log_d_measure
		self.model = model


	def run(self, particles, y, pars=None):
		
		if pars is not None:
			self.model.pars = RParam(lr=float(pars[0]), phi=float(pars[1]),
											sig=float(pars[2]), 
											N0  = 1.0,
											e0 = 0,
											T = 50)

		N = np.zeros([particles, self.model.pars.T+1]) 
		e = np.zeros([particles, self.model.pars.T+1])
		N[:, 0] = self.model.pars.N0*np.ones(particles)

		e[:, 1:] = np.random.normal(size=[particles, self.model.pars.T], scale=self.model.pars.sig)

		index = np.arange(particles)

		ll = np.zeros(self.model.pars.T)

		for i in range(1, self.model.pars.T + 1):

			N[:, i] = self.forward(self.model.pars, N[:, i-1], e[:, i-1])
			
			log_weights = self.logmeasure(self.model.pars, N[:, i], y[i-1])
			logsumexp_weights = logsumexp(log_weights)

			ll[i-1] = logsumexp_weights - np.log(particles)

			log_probs = log_weights - logsumexp_weights


			choice = np.random.choice(index, size=particles, replace=True, p = np.exp(log_probs))

			N = N[choice, :]

		return np.sum(ll)

def pmcmc(start, n_samples, y, prop_cov, pfilter, particles):
	
	# basic setup and drawing random numbers
	acceptance_rate = np.zeros([n_samples])
	test_output = np.zeros([n_samples])

	d = len(start)
	innovations = np.random.multivariate_normal(size=n_samples, mean=np.zeros([len(start)]), 
												cov=prop_cov)
	uniforms_variables = np.random.rand(n_samples)
	
	theta = np.zeros([n_samples, d])
	theta[0, :] = start

	ll = np.zeros([n_samples])

	ll[0] = pfilter.run(particles, y, pars=start)
	
	t = trange(1, n_samples, desc='Bar desc', leave=True, ncols=150)
	for i in t:
		
		# Propose new parameters
		theta_prop = theta[i - 1, :] + innovations[i, :] 
		
		# Check that all values are in prior range to avoid overflow
		if any(theta_prop < [3, 0, 0]) or any(theta_prop > [8, 20, 0.6]):
			acc = -math.inf
			test_output[i] = 1
			# print(theta_prop)
		else:
			loss_prop = pfilter.run(particles, y, pars=theta_prop)
			
			# Acceptance probability
			acc = loss_prop - ll[i-1]
		
		if np.log(uniforms_variables[i]) <= acc:
			
			theta[i, :] = theta_prop
			ll[i] = loss_prop
			acceptance_rate[i-1] = 1
		else:
			theta[i, :] = theta[i-1,:]
			ll[i] = ll[i-1]
		
		t.set_description("Running MCMC")
		t.refresh() # to show immediately the update
		t.set_postfix({"Acc.:": np.mean(acceptance_rate[0:i]), "test: ": np.mean(test_output)})

	#print("Acc rate: {}".format(novel_count/n_mcmc_samples))
	print(np.mean(test_output))
	return theta, ll



if __name__=="__main__":

	from matplotlib import pyplot as plt
	plt.style.use("classic")
	import seaborn as sns
	sns.set()
	import pandas as pd
	import pickle
	import os
	import arviz as az
	#sns.set_theme()


	model = RickerModel()
	steps = 50
	y, N = model.simulate(T=steps, seed=0)
	df = pd.DataFrame({'y': y})
	df.to_csv("ricker_{}.csv".format(steps))

	pfilter = BootstrapPF(r_forward, r_measure, log_d_measure, model)
	ll = np.zeros(1000)

	for i in range(1000):
		ll[i] = pfilter.run(particles=280, y=y, pars=np.array([3.9, 10.3, 0.43]))
	
	print("std of log-likelihood estimator")
	print(np.std(ll))

	samples = 10000
	warum_up_run, _ = pmcmc(start=np.array([4., 10., 0.3]), n_samples=samples, y=y, prop_cov=2.1/np.sqrt(3)*np.diag([0.05, 0.3, 0.08]), pfilter=pfilter, particles=280)

	print(np.mean(warum_up_run, axis=0))
	print(np.std(warum_up_run, axis=0))
	post_cov = np.cov(np.transpose(warum_up_run))
	print(post_cov)

	samples = 1000000
	mcmc_samples, ll = pmcmc(start=np.array([4., 10., 0.3]), n_samples=samples, y=y, prop_cov=2.1/np.sqrt(3)*post_cov, pfilter=pfilter, particles=280)

	print("likelihood as stationarity")
	print(np.std(ll))
	print(np.mean(mcmc_samples, axis=0))
	print(np.std(mcmc_samples, axis=0))

	thinning = np.arange(0, samples, 10)
	plot_samples = pd.DataFrame(mcmc_samples[thinning, :])

	PATH = 'ricker_truth_{}.pickle'.format(samples)
	
	if os.path.exists(PATH):
		file_h = open(PATH, 'rb') 
		plot_samples = pickle.load(file_h)
	else:
		file_h = open(PATH, 'wb') 
		pickle.dump(plot_samples, file_h)

	plot_samples= plot_samples.rename(columns={0: "r", 1: "phi", 2: "sigma"})
	print(plot_samples.head())
	print(az.ess(plot_samples['r'].values))
	g = sns.PairGrid(plot_samples)
	g = g.map_diag(sns.kdeplot, shade=True, color="r")
	g.map_lower(sns.kdeplot, cmap = 'Reds', fill=True)
	g.savefig("pmcmc_{}.png".format(samples))
