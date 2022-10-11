from grakel.kernels import WeisfeilerLehman, VertexHistogram
from numba import njit
import numpy as np
from scipy.ndimage import shift
import sigkernel
from sklearn.metrics.pairwise import euclidean_distances
import torch

from sabc.utils import _mmd


class WL_t_Kernel:
	
	def __init__(self,
				 y=None,
				 base_t_kernel="gaussian",
				 n_iter=2,
				 base_graph_kernel=VertexHistogram,
				 normalise=True):

		self.n_iter = n_iter
		# ASSUMES y is shape (length_Y, dim_Y, dim_Y)
		self._y = self._add_batch_channel(y)
		self.base_graph_kernel = base_graph_kernel
		self.base_graph_kernel_name = None
		if base_t_kernel == "gaussian":
			print('gaussian time kernel')
			# Find observation times for graphs
			ty = self._find_times(self._y)
			print(ty, ty.size())
			# Use median heuristic on observation times
			sigma = np.median(euclidean_distances(ty.reshape(-1,1)))
			print(sigma)
			# Initialise RBF kernel
			self.base_t_kernel = sigkernel.RBFKernel(sigma=sigma)
			self.base_t_kernel_name = base_t_kernel
		else:	
			self.base_t_kernel = base_t_kernel
		self.normalise = normalise
		self.wl = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=base_graph_kernel, 
								   normalize=normalise)
		self._flag = False

	def _find_times(self, X):
	
		"""
		Assumes X is shape (batch_X, length_X, dim_X, dim_X)
		"""

		tX = torch.from_numpy(np.tile(np.linspace(0, 1, X.shape[1]),
									   (X.shape[0],1,1)
									  ).transpose((0,2,1))
							  )
		return tX

	def _add_batch_channel(self, X):

		return np.expand_dims(X, axis=0)

	def Gram_matrix(self, X, Y=None):

		"""
		***ASSUMES DATA IN X AND Y EQUALLY SPACED ON INTERVAL [0,1].***

		Input:
		- X:	(batch_X, length_X, dim_X, dim_X). A collection of size batch_X of
				sequences of length length_X of adjacency matrices for dim_X nodes
		- Y:	(batch_Y, length_Y, dim_Y, dim_Y). As above
		
		Output:
		- G:	Gram matrix (batch_X, batch_Y, length_X, length_Y), such that element
				(i, j, s, t) is k(X^i_s, Y^j_t)
		"""

		## GRAPH KERNEL ##
		labels = range(X.shape[3])
		# reshape into array of adjacency matrices (for each timestep, for each
		# item within the batch)
		_X = X.reshape(-1, X.shape[2], X.shape[3]) 
		# Create iterable of adjacency matrices, and dictionary
		# with degree of each node within that network - these are then
		# used as node attributes when calculating the kernel
		iterable = [[x.numpy(), dict(zip(labels, [1]*len(labels)# x.sum(dim=0).tolist())
									))] for x in _X]
		if Y is None:
			# no comparison set given, just get the gram matrix given the data
			G = self.wl.fit_transform(iterable)
		else:
			# if Y is given, fit kernel on X then calculate gram matrix 
			# comparing with Y ()
			self.wl.fit(iterable)
			_Y = Y.reshape(-1, Y.shape[2], Y.shape[3])
			iterable = [[x.numpy(), dict(zip(labels, [1]*len(labels)#x.sum(dim=0).tolist()
											))] for x in _Y]
			G = self.wl.transform(iterable)
		
		# Correct the shape - will output as 
		# batch_X/Y*length_X/Y x batch_X*length_X
		
		N = G.shape[1]//X.shape[0] # recover length_X
		
		# reshape to batch_X x batch_X/Y*length_X/Y x length X
		G = G.reshape(G.shape[0],-1,N).swapaxes(0,1) 
		# finally reshape into desired output shape
		if Y is None:
			G = G.reshape(X.shape[0], X.shape[0], X.shape[1], X.shape[1])
		else:
			G = G.reshape(X.shape[0], Y.shape[0], X.shape[1], Y.shape[1])

		G = torch.as_tensor(G)

		## TIME KERNEL ##		
		if self.base_t_kernel is None:
			return G

		elif self.base_t_kernel_name == "gaussian":
			tX = self._find_times(X)
			if not (Y is None):
				tY = self._find_times(Y)
				t_gram = self.base_t_kernel.Gram_matrix(tX, tY)
			else:	
				t_gram = self.base_t_kernel.Gram_matrix(tX, tX)
			G = G * t_gram

		else:
			raise NotImplementedError("Haven't thought out other options yet")
	
		return G

	def batch_kernel(self, X, Y=None):

		G = self.Gram_matrix(X, Y)

		return G[0, ...]

	def Gram_matrix_naive(self, X, Y):
		
		"""
		Input:
		- X:	(batch_X, length_X, dim_X, dim_X). A collection of size batch_X of
				sequences of length length_X of adjacency matrices for dim_X nodes
		- Y:	(batch_Y, length_Y, dim_Y, dim_Y). As above
		
		Output:
		- G:	Gram matrix (batch_X, batch_Y, length_X, length_Y), such that element
				(i, j, s, t) is k(X^i_s, Y^j_t)
		"""

		# TODO: just flatten X & Y so that it computes everything in batch, then just reshape
		gram_mat = np.empty((X.shape[0], X.shape[0], X.shape[1], X.shape[1]))
		labels = range(X.shape[3])
		node_labels = dict(zip(labels, labels))
		# This computes the gram matrix between all length_X graphs
		for i in range(X.shape[0]):
			# List of sublists, where each sublist contains (a) array of shape 
			# (length_of_sequence, n_nodes, n_nodes) (e.g a graph growing over 
			# length_of_sequence steps and consisting of a total of n_nodes nodes
			# (eventually)) and (b) a dictionary mapping from matrix column id to
			# node label
			iterable = [[x.numpy(), node_labels] for x in X[i]]
			# Need to compute kernel between all pairs of items in the sequence
			G = self.wl.fit_transform(iterable)
			gram_mat[i, i, :, :] = G
			for j in range(i+1, X.shape[0]):
				# This will be of shape (length_X[j], length_X[i]), due to shape conventions in sklearn
				gmij = self.wl.transform([[x.numpy(), node_labels] for x in X[j]])
				gram_mat[j, i, :, :] = gmij
				# So need to transpose to get into shape (length_X[i], length_X[j])
				gmij = gmij.T
				gram_mat[i, j, :, :] = gmij
		return torch.as_tensor(gram_mat)
