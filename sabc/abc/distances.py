import numpy as np
import ot
import sigkernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import torch
from tqdm import tqdm

from sabc.utils import _mmd, kernels


def _print_number(fl, n_digits=4):
	fl = str(fl)
	return fl[:min([n_digits, len(fl)])]


class SignatureDistance:

	def __init__(self, y, static_kernel=None, dyadic_order=1, lead_lag=False,
				 add_time=False, rescale=None, mean=0., cumsum=False, order=-1):

		"""
		y is observation; static_kernel is the static kernel to be used in
		signature kernel computations and must have the same methods as
		sigkernel.RBFKernel.

		Assumes y is of shape (T,) or (T, C) where T is length and C is number
		of channels

		If order == -1, the full untruncated signature kernel is used as in
		Salvi et al (2020). Otherwise, order >= 1 means truncated signature
		kernel with specified order will be used.
		"""

		# Set bool transform variables
		self._at = add_time
		self._ll = lead_lag
		self._rescale = rescale
		self._cs = cumsum
		self._mean = mean
		self._do = dyadic_order
		self._sk = static_kernel
		self._order = order
		self.skernel = None

		# Store observation
		self._y = y
		self._obs = None
		self._kyy = None
		self._gaussian = False

		if not self._rescale is None:
			self._obs = self._preprocess(y)
			# Set the static kernel
		if (self._sk is None):
			# Compute scale parameter for Gaussian kernel
			#sigma = _mmd._get_rbf_bandwidth(y)
			self._gaussian = True
			if not (self._obs is None):
				assert len(self._obs.size()) == 3, "Obs should be 3D"
				sigma = np.median(euclidean_distances(self._obs[0], self._obs[0]))
				# Create Gaussian static kernel
				self._sk = sigkernel.RBFKernel(sigma=sigma)
				if self._order >= 1:
					self.skernel = kernels.TruncatedSigKernel(self._sk, self._order)
				else:
					# Create signature kernel object
					self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)
				# Precompute <y,y> term
				self._kyy = self.skernel.compute_kernel(self._obs, self._obs)
		elif not (self._sk is None):	
			if self._order >= 1:
				print("Using truncated signature kernel")
				self.skernel = kernels.TruncatedSigKernel(self._sk, self._order)
			else:
				print("Using untruncated signature kernel")
				# Create signature kernel object
				self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)
			if not (self._obs is None):
				# Precompute <y,y> term
				self._kyy = self.skernel.compute_kernel(self._obs, self._obs)
		
	def set_static_kernel(self, kernel):

		self._sk = kernel
		if self._order >= 1:
			self.skernel = kernels.TruncatedSigKernel(self._sk, self._order)
		else:
			# Create signature kernel object
			self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)
		self._kyy = self.skernel.compute_kernel(self._obs, self._obs)

	def _strip_time(self, x):

		return x[..., :-1]

	def _lead_lag(self, x):

		repeat_x = x#np.repeat(x, 2, axis=1)
		z = np.concatenate((repeat_x[:, :-1, :], repeat_x[:, 1:, :]), axis=-1)
		return z

	def _add_time(self, x):

		ts = np.arange(x.shape[1]) / (x.shape[1] - 1)
		ts = np.repeat(ts.reshape(1, ts.size, 1), x.shape[0], axis=0)
		x = np.concatenate((x, ts), axis=-1)
		return x

	def set_dyadic_order(self, dyadic_order):

		assert isinstance(dyadic_order, int), "Must be an integer >= 0"
		if dyadic_order < 0:
			raise ValueError("Must be an integer >= 0")
		self._do = dyadic_order

	def _preprocess(self, x):

		if self._cs:
			x[:, :-1] = np.cumsum(x[:, :-1], axis=0)
		x = np.expand_dims(x, axis=0)
		if self._ll:
			x = self._strip_time(x)
			x = self._lead_lag(x)
			x = self._add_time(x)
		if not self._ll and self._at:
			x = self._add_time(x)
		x = x - self._mean
		x /= self._rescale
		return torch.from_numpy(x)

	def update_rescale(self, rescale):

		self._rescale = rescale

	def update_mean(self, mean):

		self._mean = mean

	def update_obs(self, y):

		self._y = y
		self._obs = self._preprocess(y)
		if self._gaussian:
			sigma = np.median(euclidean_distances(self._obs[0], self._obs[0]))
			# Create Gaussian static kernel
			self._sk = sigkernel.RBFKernel(sigma=sigma)
			# Create signature kernel object
			if self._order >= 1:
				self.skernel = kernels.TruncatedSigKernel(self._sk, self._order)
			else:
				# Create signature kernel object
				self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)
		self._kyy = self.skernel.compute_kernel(self._obs, self._obs)

	def train(self, X):
	
		if self._cs:
			X[:, :, :-1] = np.cumsum(X[:, :, :-1], axis=1)
		if self._ll:
			X = self._strip_time(X)
			X = self._lead_lag(X)
			X = self._add_time(X)
		if not self._ll and self._at:
			X = self._add_time(X)
		mins = X.min(axis=1)
		maxs = X.max(axis=1)
		rang = (maxs - mins).mean(axis=0)
		mean = np.mean(X, axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]
		self.update_rescale(rang)
		self.update_mean(mean)

		X = X - mean
		X /= self._rescale
		self.update_obs(self._y)
		return self

	def compute_distance(self, x):

		"""
		Assumes x is of shape (T,) or (T, C), thus applying expand_dims to
		first dimension
		"""

		x = self._preprocess(x)
		kxx = self.skernel.compute_kernel(x, x)
		kxy = self.skernel.compute_kernel(x, self._obs)
		#return self._kyy + kxx - 2*kxy
		return 2.*(1. - kxy / np.sqrt(self._kyy * kxx))


class SignatureRegressionDistance:

	def __init__(self, y, static_kernel=None, dyadic_order=1, lead_lag=False,
				 add_time=False, rescale=None, mean=0., cumsum=False, order=-1,
				 regressor='srd'):

		"""
		y is observation; static_kernel is the static kernel to be used in
		signature kernel computations and must have the same methods as
		sigkernel.RBFKernel.

		Assumes y is of shape (T,) or (T, C) where T is length and C is number
		of channels
		"""

		# Set bool transform variables
		self._at = add_time
		self._ll = lead_lag
		self._rescale = rescale
		self._cs = cumsum
		self._mean = mean
		self._do = dyadic_order
		self._sk = static_kernel
		self._order = order
		self.skernel = None
		self._regressor = regressor

		# Store observation
		self._y = y
		self._obs = None
		self._gaussian = False

		if not self._rescale is None:
			self._obs = self._preprocess(y)
			# Set the static kernel
		if (self._sk is None):
			self._gaussian = True
			if not (self._obs is None):
				assert len(self._obs.size()) == 3, "Obs should be 3D"
				# Compute scale parameter for Gaussian kernel
				sigma = np.median(euclidean_distances(self._obs[0], self._obs[0]))
				# Create Gaussian static kernel
				self._sk = sigkernel.RBFKernel(sigma=sigma)
				if self._order >= 1:
					self.skernel = kernels.TruncatedSigKernel(self._sk, self._order)
				else:
					# Create signature kernel object
					self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)
		elif not (self._sk is None):	
			if self._order >= 1:
				print("Using truncated signature kernel")
				self.skernel = kernels.TruncatedSigKernel(self._sk, self._order)
			else:
				print("Using untruncated signature kernel")
				# Create signature kernel object
				self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)
		
		# Initialise kernel ridge regression attribute
		self._kr = None
		self._sobs = None
		self._X = None

	def update_rescale(self, rescale):

		self._rescale = rescale

	def update_mean(self, mean):

		self._mean = mean

	def update_obs(self, y):

		self._y = y
		self._obs = self._preprocess(y)
		if self._gaussian:
			sigma = np.median(euclidean_distances(self._obs[0], self._obs[0]))
			self._sk = sigkernel.RBFKernel(sigma=sigma)
			if self._order >= 1:
				self.skernel = kernels.TruncatedSigKernel(self._sk, self._order)
			else:
				self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)

	def _update_skernel(self):
	
		self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)
		
	def set_static_kernel(self, kernel):

		self._sk = kernel
		self._update_skernel()

	def set_dyadic_order(self, dyadic_order):

		assert isinstance(dyadic_order, int), "Must be an integer >= 0"
		if dyadic_order < 0:
			raise ValueError("Must be an integer >= 0")
		self._do = dyadic_order
		self._update_skernel()

	def _preprocess(self, x):

		if self._cs:
			x[:, :-1] = np.cumsum(x[:, :-1], axis=0)
		x = np.expand_dims(x, axis=0)
		x = x - self._mean
		x = sigkernel.transform(x, at=self._at, ll=self._ll,
								scale=1./self._rescale)
		return torch.from_numpy(x)

	def train(self, X, thetas, alphas=None, theta_transform=None):

		"""
		Train kernel ridge regression model.

		Input:
		- X:		np.array of shape (B, T, C), where B is batch size, T is
					length, C is number of channels
		- thetas:	np.array of shape (B, D), where B is as above and D is the
					parameter dimension
		"""

		if self._cs:
			X[:, :, :-1] = np.cumsum(X[:, :, :-1], axis=1)
		mins = X.min(axis=1)
		maxs = X.max(axis=1)
		rang = (maxs - mins).mean(axis=0)
		mean = np.mean(X, axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]
		self.update_rescale(rang)
		self.update_mean(mean)
		self.update_obs(self._y)
		X = sigkernel.transform(X - self._mean, at=self._at, ll=self._ll, 
								scale=1./self._rescale)
		X = torch.from_numpy(X)

		# Transform thetas as desired
		if not theta_transform is None:
			thetas = theta_transform(thetas)

		best_score = -1*float("inf")
		# Default search space. TODO: make this customisable
		iterator = tqdm([10**(-i+2) for i in range(6)], desc=("Tuning signature "+
						"kernel RBF hyperparameter"))
		for sigma in iterator:
			# Specify the static kernel 
			static_kernel = sigkernel.RBFKernel(sigma=sigma)

			# Initialize the corresponding signature kernel
			signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=self._do)

			# Gram matrix train
			G_train = signature_kernel.compute_Gram(X, X, sym=True).numpy()
			if alphas is None:
				alphas = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100.]
			if self._regressor == 'srd':
				parameters = {'alpha':alphas}
				regressor = KernelRidge(kernel='precomputed', gamma=0.1)
				#krr.solver = 'svd'
			elif self._regressor == "svd":
				parameters = {'C':alphas}
				regressor = SVR(kernel='precomputed', gamma=0.1)
			regressor_cv = GridSearchCV(estimator=regressor, param_grid=parameters,
										cv=5, n_jobs=-1, error_score='raise')
			regressor_cv.fit(G_train, thetas)
			if regressor_cv.best_score_ > best_score:
				best_reg = regressor_cv
				best_score = regressor_cv.best_score_
				best_sigma = sigma
				best_alpha = regressor_cv.best_params_['alpha']
			iterator.set_postfix({"s,scr":(sigma, _print_number(regressor_cv.best_score_)), 
								  "Best":(best_sigma, _print_number(best_score)),
								  "alpha":best_alpha})
		self._kr, sigma = best_reg.best_estimator_, best_sigma
		self._sk = sigkernel.RBFKernel(sigma=sigma)
		self.skernel = sigkernel.SigKernel(self._sk, dyadic_order=self._do)

		# Predict observation summaries
		G_obs = self.skernel.compute_Gram(self._obs, X, sym=False).numpy()
		self._sobs = self._kr.predict(G_obs)
		self._X = X
		return self

	def compute_distance(self, x):

		"""
		Assumes x is of shape (T,) or (T, C), thus applying expand_dims to
		first dimension
		"""

		if self._kr is None:
			raise RuntimeError("Kernel regression model not trained")
		x = self._preprocess(x)
		G_test = self.skernel.compute_Gram(x, self._X, sym=False).numpy()
		sx = self._kr.predict(G_test)
		return np.sum((sx - self._sobs)**2)


class SemiAutomaticDistance:

	def __init__(self, y, summary_statistics, remove_time=False, strip_bp=False):

		"""
		y is observation; summary_statistics is a callable which consumes
		a batch of datasets of shape (B, T, C) -- where B is batch size, T is
		length, and C is number of channels -- and returns summary statistics
		of that batch, which are then used as predictors in the semi-automatic
		regression task

		Assumes y is of shape (T,) or (T, C) where T is length and C is number
		of channels
		"""

		self._ss = summary_statistics
		self._y = y
		self._rt = remove_time
		self._sbp = strip_bp
		self._obs = self._preprocess(y)
		self._lr = None
		self._sobs = None

	def _remove_time(self, x):

		"""
		Removes time channel from numpy array of data x
		"""

		if self._rt:
			return x[..., :-1]
		else:
			return x

	def _strip_bp(self, x):

		if self._sbp:
			x = x[:, 1:, ...]
		return x

	def _preprocess(self, x, no_batch_dim=True):

		x = self._remove_time(x)
		if no_batch_dim:
			x = np.expand_dims(x, axis=0)
		x = self._strip_bp(x)
		return self._ss(x)

	def train(self, X, thetas, alphas=None, theta_transform=None):

		"""
		Train linear regression model.

		Input:
		- X:		np.array of shape (B, T, C), where B is batch size, T is
					length, C is number of channels
		- thetas:	np.array of shape (B, D), where B is as above and D is the
					parameter dimension
		"""

		# Overwrite theta scaling
		if not theta_transform is None:
			thetas = theta_transform(thetas)

		# Covert X into initial summary statistics
		sX = self._preprocess(X, no_batch_dim=False)

		self._lr = LinearRegression()
		self._lr.fit(sX, thetas)

		# Predict observation summaries
		self._sobs = self._lr.predict(self._obs)
		return self

	def compute_distance(self, x):

		"""
		Assumes x is of shape (T,) or (T, C), thus applying expand_dims to
		first dimension
		"""

		if self._lr is None:
			raise RuntimeError("Linear regression model not trained")
		x = self._preprocess(x)
		sx = self._lr.predict(x)
		return np.sum((sx - self._sobs)**2)


class MMDDistance:

	def __init__(self, y, base_kernel=None, cumsum=False, remove_time=True, unsqueeze=False):

		"""
		y is observation; base_kernel is the kernel to be used in MMD
		computations, and it should have a .compyute_Gram method.

		Assumes y is of shape (T,) or (T, C) where T is length and C is number
		of channels

		preprocess should be a callable taking in whatever dataset and
		returning it in the form required by the base kernel
		"""

		# TODO: check that this is happy with non-Gaussian kernel

		self._cs = cumsum
		self._rt = remove_time
		self._us = unsqueeze
		# Preprocess observation 
		self._obs = self._preprocess(y)
		# Set the static kernel - assumes Gaussian RBF if none specified
		if base_kernel is None:
			# Compute scale parameter for Gaussian kernel
			sigma = 2*_mmd._get_rbf_bandwidth(self._obs)**2
			# Create Gaussian static kernel
			base_kernel = sigkernel.RBFKernel(sigma=sigma)
		self.set_base_kernel(base_kernel)

		if not torch.is_tensor(self._obs):
			self._obs = torch.from_numpy(self._obs)

		# Precompute E[Y,Y_] term
		self._EYY_ = self._mmd_term(self._obs, self._obs, same=True)

	def _remove_time(self, x):

		"""
		Removes time channel from numpy array of data x
		"""

		if not self._rt:
			return x
		return x[..., :-1]

	def _preprocess(self, x):

		x = self._remove_time(x)
		if self._cs:
			x = np.cumsum(x, axis=0)
		return x
		
	def set_base_kernel(self, kernel):

		self._base = kernel

	def _mmd_term(self, x, y, same=False):

		#zero_diag = False
		if not torch.is_tensor(x):
			x = torch.from_numpy(x)
		if self._us:
			x = x.unsqueeze(0)
		if not torch.is_tensor(y):
			y = torch.from_numpy(y)
		if self._us:
			y = y.unsqueeze(0)
		_gram = self._base.Gram_matrix(x, y).numpy()[0,0,...]
		if same:
			_gram[np.diag_indices(_gram.shape[0])] = 0.
		return _gram.mean()	

	def compute_distance(self, x):

		"""
		Assumes x is of shape (T,) or (T, C)
		"""

		x = self._preprocess(x)
		EXX_ = self._mmd_term(x, x, same=True)
		EXY  = self._mmd_term(x, self._obs, same=False)
		return self._EYY_ + EXX_ - 2*EXY


class WassersteinDistance:

	def __init__(self, y, add_time=False, lmbda=None, cumsum=False, lead_lag=False):

		"""
		y is observation; base_kernel is the kernel to be used in MMD
		computations, and it should have a .compyute_Gram method.

		Assumes y is of shape (T,) or (T, C) where T is length and C is number
		of channels

		add_time: True if add time channel before computing distance. Assumes
		the time channel is in the LAST (-1) position of the LAST dimension

		lmbda: float determining relative weight assigned to horizontal and
		vertical distances. See paper
		"""

		# Set preprocess method
		self._at = add_time
		self._lambda = lmbda
		self._cs = cumsum
		self._ll = lead_lag
		self._y = y
		self._obs = y
		if self._cs:
			self._obs[:, :-1] = np.cumsum(self._obs[:, :-1], axis=0)
		if not (self._lambda is None):
			self._obs[:, -1] *= self._lambda
		if self._ll:
			self._obs = self._strip_time(self._obs)
			self._obs = self._lead_lag(self._obs)
			self._obs = self._add_time(self._obs)

	def _strip_time(self, x):

		return x[..., :-1]

	def _lead_lag(self, x):

		remove_batch_index = False
		if len(x.shape) == 2:
			remove_batch_index = True
			x = x[np.newaxis, ...]
		repeat_x = x#np.repeat(x, 2, axis=1)
		z = np.concatenate((repeat_x[:, :-1, :], repeat_x[:, 1:, :]), axis=-1)
		if remove_batch_index:
			z = z[0]
		return z

	def _add_time(self, x):

		remove_batch_index = False
		if len(x.shape) == 2:
			remove_batch_index = True
			x = x[np.newaxis, ...]
		ts = np.arange(x.shape[1]) / (x.shape[1] - 1)
		ts = np.repeat(ts.reshape(1, ts.size, 1), x.shape[0], axis=0)
		x = np.concatenate((x, ts), axis=-1)
		if remove_batch_index:
			x = x[0]
		return x

	def train(self, X):

		"""
		Assumes all arrays equal length so can contain in the same numpy
		array. Should adjust so that it can deal with continuous time models
		"""

		if isinstance(X, np.ndarray):
			if self._cs:
				X[:, :, :-1] = np.cumsum(X[:, :, :-1], axis=1)
			if self._ll:
				X = self._strip_time(X)
				X = self._lead_lag(X)
				X = self._add_time(X)
			# Need to figure out how to get cumsum in here
			mins = X.min(axis=1)
			maxs = X.max(axis=1)
			# This is channel-wise range
			rang = (maxs - mins).mean(axis=0)
			# Get "horizontal" range
			T = rang[-1]
			# Now max vertical range
			# TODO: should this be the magnitude of the array?
			V = np.sqrt(np.sum(rang[:-1]**2))
		elif isinstance(X, list):
			# Loop through and find range
			# Then again, perhaps overkill since for GSE we know lmbda
			pass

		self._lambda = float(V / T)	
		self._obs[:, -1] *= self._lambda
		return self
		
	def compute_distance(self, x):

		"""
		Assumes x is of shape (T,) or (T, C)
		"""

		# Between non-time components
		if self._cs:
			x[:, :-1] = np.cumsum(x[:, :-1], axis=0)
		if self._ll:
			x = self._strip_time(x)
			x = self._lead_lag(x)
			x = self._add_time(x)
		x[:, -1] *= self._lambda
		M = ot.dist(x[:, :-1], self._obs[:, :-1], metric="euclidean")
		# Between time components
		O = ot.dist(x[:, -1:], self._obs[:, -1:], metric="euclidean")
		# Full distance is < Euclidean on non-time > + lambda * | t - s |
		D = M + O
		gamma, log = ot.emd([], [], D, log=True)

		return log['cost']
