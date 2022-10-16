import errno
import numpy as np
import os
from sabc.models import GBM, Ricker, GSE, BH, ZMN
from sabc.utils import distributions
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.tsa.stattools as sttsa
import time

this_dir = os.path.dirname(os.path.realpath(__file__))
task2task = {"gbm":"GBM",
			 "ricker":"Ricker",
			 "gse":"GSE",
			 "bh":"BH",
			 "bhn":"BH",
			 "zmn":"ZMN"}

def _name2T(name):

	elif name == "gbm":
		T = 100
	elif name == "ricker":
		T = 50
	elif name in ["bh", "bhn"]:
		T = 100
	elif name == "gse":
		T = None
	elif name == "zmn":
		T = 25
	return T

def _load_simulator(task_name):

	elif task_name == "bh":
		model = BH.Model()
	elif task_name == "bhn":
		model = BH.Model(beta=10.)
	elif task_name == "gbm":
		model = GBM.Model()
	elif task_name == "ricker":
		model = Ricker.Model()
	elif task_name == "gse":
		model = GSE.Model()
	elif task_name == "zmn":
		model = ZMN.Model(N=20)

	# Everything is basepoint-ed by default
	def simulator(pars):
		return model.simulate(pars=pars, T=_name2T(task_name))

	return simulator

def _load_prior(task_name):

	elif task_name == "bh":
		prior = distributions.BoxUniform(low =[ 0., 0., 0.,-1.],
										 high=[ 1., 1., 1., 0.])
	elif task_name == "bhn":
		prior = distributions.BoxUniform(low =[-1.,-1., 0., 0.],
										 high=[ 0., 0., 1., 1.])
	elif task_name == "gbm":
		prior = distributions.BoxUniform(low =[-1.,0.2],
										 high=[ 1.,2. ])
	elif task_name == "ricker":
		prior = distributions.BoxUniform(low =[ 3. , 0. , 0. ],
										 high=[ 8. ,20. , 0.6])
	elif task_name == "gse":
		prior = distributions.GSEBoxGamma(lmbdas=[0.1,0.2],
								  		  nus	=[2. ,0.5])
	elif task_name == "zmn":
		prior = distributions.BoxUniform(low = [0., 0.],
										 high = [1.,1.])
	return prior
	
def _load_dataset(task_name):
	
	elif task_name == "bh":
		y = np.loadtxt(os.path.join(this_dir, "../data/BH/obs.txt"))
		ts = np.arange(y.size).reshape(-1,1)
		ts = ts / float(np.max(ts))
		y = np.concatenate((y.reshape(-1,1), ts), axis=-1)
	elif task_name == "bhn":
		y = np.loadtxt(os.path.join(this_dir, "../data/BHN/obs.txt"))
		ts = np.arange(y.size).reshape(-1,1)
		ts = ts / float(np.max(ts))
		y = np.concatenate((y.reshape(-1,1), ts), axis=-1)
	elif task_name == "gbm":
		y = np.loadtxt(os.path.join(this_dir, "../data/GBM/obs.txt"))
		ts = np.arange(y.size).reshape(-1,1)
		ts = ts / float(np.max(ts))
		y = np.concatenate((y.reshape(-1,1), ts), axis=-1)
	elif task_name == "ricker":
		y = np.loadtxt(os.path.join(this_dir, "../data/Ricker/obs.txt"))
		ts = np.arange(y.size).reshape(-1,1)
		ts = ts / float(np.max(ts))
		y = np.concatenate((y.reshape(-1,1), ts), axis=-1)
	elif task_name == "gse":
		y = np.loadtxt(os.path.join(this_dir, "../data/GSE/obs.txt"))
	elif task_name == "zmn":
		y = np.load(os.path.join(this_dir, "../data/ZMN/obs_cumsum.npy"))
	return y

def _load_true_pars(task_name):
	
	elif task_name == "bh":
		theta = np.array([0.9,0.2,0.9,-0.2])
	elif task_name == "bhn":
		theta = np.array([-0.7,-0.4,0.5,0.3])
	elif task_name == "gbm":
		theta = np.array([0.2,0.5])
	elif task_name == "ricker":
		theta = np.array([4.,10.,.3])
	elif task_name == "gse":
		theta = np.array([1e-2,1e-1])
	elif task_name == "zmn":
		theta = np.array([0.4, 0.7])
	return theta

def _load_summariser(task_name):

	elif task_name in ["bh", "bhn"]:

		def summariser(x):
			x = x[:, :, 0]
			oss = x[:, ::10]
			z = oss.copy()
			z = np.concatenate((z, oss**2), axis=-1)
			return z

	if task_name == "gbm":

		def summariser(x):
			x = np.log(x[:, :, 0])
			x = np.diff(x, axis=-1)
			var = np.var(x, axis=1).reshape(-1,1)
			rho1s, rho2s = [], []
			for _x in x:
				rho1, rho2 = sm.tsa.acf(_x, nlags=2)[1:]
				rho1s.append(rho1)
				rho2s.append(rho2)
			rho1s, rho2s = np.array(rho1s).reshape(-1,1), np.array(rho2s).reshape(-1,1)
			x = np.concatenate((var, rho1s, rho2s), axis=1)
			z = x.copy()
			for i in range(2, 5):
				x = np.concatenate((x, z**i), axis=-1)
			return x

	elif task_name == "ricker":

		y = _load_dataset(task_name)[1:, 0]
		diff_y = np.diff(y)
		sorted_diff_y = np.sort(diff_y).reshape(-1, 1)

		def summariser(x):
			x = x[..., 0]
			# Number of zeros in each series
			n0s = np.sum(x == 0, axis=1).reshape(-1,1)
			# Mean values
			mns = np.mean(x, axis=1).reshape(-1,1)
			ac5, coefs1, coefs2 = [], [], []
			for _x in x:

				# Autocovariances to lag 5
				ac5.append(sttsa.acovf(_x, fft=False, nlag=5))

				# Regression thing
				target = np.power(_x[1:], 0.3)
				regressors = np.concatenate((np.power(_x[:-1], 0.3).reshape(-1,1),
											 np.power(_x[:-1], 0.6).reshape(-1,1)), axis=1)
				coefs1.append(LinearRegression().fit(regressors, target).coef_)

				# Cubic regression thing
				target = np.sort(np.diff(_x))
				coefs2.append(LinearRegression().fit(sorted_diff_y, target).coef_)

			ac5 = np.stack(ac5)
			coefs1 = np.stack(coefs1)
			coefs2 = np.stack(coefs2)
			to_return = np.concatenate((n0s, mns, ac5, coefs1, coefs2), axis=1)
			return to_return

	elif task_name in ["gse", "zmn"]:

		summariser = None

	return summariser

def _load_theta_transformer(task_name):

	if task_name == "gse":
		return lambda x: x
	elif task_name == "bh":
		high, low = np.array([1.,1.,1.,0.]), np.array([0.,0.,0.,-1.])
	elif task_name == "bhn":
		high, low = np.array([0.,0.,1.,1.]), np.array([-1.,-1.,0.,0.])
	elif task_name == "gbm":
		high, low = np.array([1.,2.]), np.array([-1,0.2])
	elif task_name == "ricker":
		high, low = np.array([8.,20.,0.6]), np.array([3.,0.,0.])
	elif task_name == "zmn":
		high, low = np.array([1.,1.]), np.array([0.,0.])
	rnge = high - low
	fun = lambda x: (x - low)/rnge
	return fun

def load_task(task):

	try:
		task_var = eval(task2task[task])
	except:
		raise NotImplementedError("Task not recognised")
	prior = _load_prior(task)
	simulator = _load_simulator(task)
	obs  = _load_dataset(task)
	summariser = _load_summariser(task)
	theta_transform = _load_theta_transformer(task)

	return simulator, prior, obs, summariser, theta_transform

def prep_outloc(args, seed):

	# Ensure directory exists -- throw error if not
	if os.path.exists(args.o):
		# If directory exists, create a subdirectory with name = timestamp
		subdir = str(time.time())
		outloc = os.path.join(args.o, subdir)
		try:
			os.mkdir(outloc)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
			pass
		# Within this subdirectory, create a job.details file showing the input
		# args to job script
		with open(os.path.join(outloc, "this.job"), "w") as fh:
			for arg in vars(args):
				fh.write("{0} {1}\n".format(arg, getattr(args, arg)))
			fh.write("seed " + str(seed) + "\n")
	else:
		raise ValueError("Output location doesn't exist -- please create the folder")

	return outloc

def save_output(samples, distances, outloc, cpu_time):

	if not samples is None:
		sample_loc = os.path.join(outloc, "samples.txt")
		np.savetxt(sample_loc, samples)
	if not distances is None:
		distances_loc = os.path.join(outloc, "distances.txt")
		np.savetxt(distances_loc, distances)
	if not cpu_time is None:
		np.savetxt(os.path.join(outloc, "cpu.time"), np.array([cpu_time]))
