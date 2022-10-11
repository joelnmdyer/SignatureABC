import numpy as np
import torch
from tqdm import trange

def run(simulator, prior, distance, n_samples, proportion, seed=None,
		verbose=True):

	"""
	Implements basic rejection ABC. Inputs are as follows:

	- simulator:	callable, should consume a parameter and return a
					stochastic output from the simulator at that parameter
	- prior:		an object with .sample() and .log_prob() methods from
					which parameter values are simulated
	- distance:		callable, should consume an output from simulator and
					return a distance from the actual data. Intention is that
					this is a method of a class which handles distance
					evaluations and the necessary transformations internally
	- n_samples:	integer >= 1 determining the number of simulations from
					the joint density over parameters and simulator output
	- proportion:	float in range [0,1] determining the proportion of the
					n_samples generated from the joint density that will
					contribute to the ABC posterior
	- seed:			optional integer argument setting the random seed for RNG

	Output:
	- samples:		np.array of shape (n_keep, d), where
						n_keep = int(proportion * n_samples)
						d = parameter dimensions
	"""

	if not (seed is None):
		np.random.seed(seed)
		torch.manual_seed(seed)
	thetas = prior.sample(n_samples)
	if torch.is_tensor(thetas):
		thetas = thetas.numpy()

	# TODO: Parallelise this with multiprocessing
	distances = []
	if verbose:
		iterator = trange(n_samples)
	else:
		iterator = range(n_samples)
	for t in iterator:
		theta = thetas[t, :]
		x = simulator(list(theta))
		dxy = distance(x)
		distances.append(float(dxy))

	# Do we want to save all output before taking the best?
	n_keep = int(proportion * n_samples)
	idx = np.argsort(distances)
	samples = thetas[idx[:n_keep]]
	return samples, np.array(distances)[idx[:n_keep]]
