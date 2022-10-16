import argparse
import numpy as np
from sabc.abc import rejection_abc, distances
from sabc.utils import io, kernels
import time
import torch

name_to_distance = {
	"wass":   distances.WassersteinDistance,
	"mmd":	  distances.MMDDistance,
	"sd":	  distances.SignatureDistance,
	"srd":	  distances.SignatureRegressionDistance,
	"sad":	  distances.SemiAutomaticDistance,
}

def generate_training_data(simulator, prior, n_train):

	train_thetas = prior.sample(n_train)
	train_xs = []
	for theta in train_thetas:
		x = simulator(theta)
		train_xs.append(x)
	train_xs = np.stack(train_xs)
	return train_xs, train_thetas

def train_regression_model(simulator,
						   prior,
						   n_train,
						   theta_transform,
						   regression_model):
	
	train_xs, train_thetas = generate_training_data(simulator, prior, n_train)
	regression_model.train(train_xs,
						   train_thetas,
						   theta_transform=theta_transform)
	return regression_model


def run(task,
		name,
		n_samples=int(1e6),
		n_train=300,
		proportion=1e-3,
		train_seed=None,
		abc_seed=None,
		verbose=True,
		depth=-1,
		leadlag=False):

	if not seed is None:
		np.random.seed(seed)
		torch.manual_seed(seed)

	# Load simulator, prior, observation, summary statistics (for SA-ABC)
	# and rescaling function for thetas in regression-based summary statistics
	out = io.load_task(task)
	simulator, prior, obs, summary_statistics, theta_transform = out
	print(obs.shape)

	# Create the thing that will compute distances for us
	distance_calculator = name_to_distance[name]

	# Train the distance calculator if necessary
	if verbose:
		print("Training distance calculators...")
	if name in ["sad", "srd"]:
		cs = False
		if name == "sad":
			sbp = task in ["ricker", "ma2"]
			dc = distance_calculator(obs, summary_statistics, remove_time=True, strip_bp=sbp)
		elif name in ["srd"]:
			if task == "ricker":
				cs = True
			# Need to update rescale parameter after simulating training set
			dc = distance_calculator(obs, None, 0, lead_lag=False, add_time=False, cumsum=cs,
									 regressor=name)
		dc = train_regression_model(simulator,
									prior,
									n_train,
									theta_transform,
									dc)

	elif name == "wass":
		if task == "gse":
			lmbda = 2.
			dc = distance_calculator(obs, False, lmbda)
		elif task == "ricker":
			dc = distance_calculator(obs, False, None, True, leadlag)
			train_xs, _ = generate_training_data(simulator, prior, n_train)
			dc = dc.train(train_xs)
		else:	
			dc = distance_calculator(obs, False, None, False, leadlag)
			train_xs, _ = generate_training_data(simulator, prior, n_train)
			dc = dc.train(train_xs)

	elif name == "mmd":
		if task in ["zmn"]:
			wl_kernel = kernels.WL_t_Kernel(obs, base_t_kernel=None)
			dc = distance_calculator(obs, base_kernel=wl_kernel,
									 remove_time=False, unsqueeze=True)
		elif task == "ricker":
			dc = distance_calculator(obs, cumsum=True, unsqueeze=True)
		else:
			dc = distance_calculator(obs, unsqueeze=True)

	elif name == "sd":
		if task == "gse":
			# Known a priori (provided you don't change the model)
			RESCALE = np.array([100., 100., 50.])
			dc = distance_calculator(obs, rescale=RESCALE, order=depth, lead_lag=leadlag)
		elif task == "ricker":	
			dc = distance_calculator(obs, None, cumsum=True, order=depth, lead_lag=leadlag)
			train_xs, _ = generate_training_data(simulator, prior, n_train)
			dc = dc.train(train_xs)
		elif task in ["zmn"]:
			wl_kernel = kernels.WL_t_Kernel(obs)
			dc = distance_calculator(obs, static_kernel=wl_kernel, rescale=1.,
									 order=depth, lead_lag=leadlag)
		else:	
			dc = distance_calculator(obs, None, order=depth, lead_lag=leadlag)
			train_xs, _ = generate_training_data(simulator, prior, n_train)
			dc = dc.train(train_xs)

	distance = dc.compute_distance

	samples, distances = rejection_abc.run(simulator,
										   prior,
										   distance,
										   n_samples,
										   proportion,
										   abc_seed,
										   verbose)

	return samples, distances

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Rejection ABC')
	parser.add_argument('--task', type=str,
						help='Name of task (simulator) to experiment with.')
	parser.add_argument('--method', type=str,
						help="""Name of distance to use. Must be in:
								["wass", "sad", "sd", "srd", "mmd"].""")
	parser.add_argument('--n', type=int, default=int(1e6),
						help='Number samples from joint.')
	parser.add_argument('--p', type=float, default=1e-3,
						help='Proportion of n to retain.')
	parser.add_argument('--m', type=int, default=300,
						help="Number of training examples.")
	parser.add_argument('--seed', type=int, nargs='+', help='Seeds for RNG.')
	parser.add_argument('--o', type=str, help="Location to dump data.")
	parser.add_argument('--d', type=int, default=-1, help="Truncation depth.")
	parser.add_argument('-v', action='store_true', help="Turns verbose mode on")
	parser.add_argument('-l', action='store_true', help="Turns lead-lag on for signatures")
	args = parser.parse_args()

	for seed in args.seed:
		# Create subdirectory and add a job.details file
		outloc = io.prep_outloc(args, seed)
		start_cpu_time = time.process_time()
		samples, distances = run(args.task,
								 args.method,
								 n_samples=args.n,
								 n_train=args.m,
								 proportion=args.p,
								 train_seed=seed,
								 abc_seed=seed+100,
								 verbose=args.v,
								 depth=args.d,
								 leadlag=args.l)
		end_cpu_time = time.process_time()
		# Save samples and distances to a single csv file
		io.save_output(samples, distances, outloc, end_cpu_time - start_cpu_time)
