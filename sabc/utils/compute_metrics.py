import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ot
import ot.sliced

matplotlib.rc('text', usetex=True)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})
plt.rcParams.update({
	'text.latex.preamble':r"\usepackage{amsmath}"+"\n"+r"\usepackage{bm}"
})
#matplotlib.rcParams.update({
#    'pgf.texsystem': "pdflatex",
#    'font.family': 'serif',
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#    "pgf.preamble": "\n".join([
#         r"\usepackage{bm}",
#    ]),
#    'font.serif': ["Computer Modern Roman"],
    #'text.latex.preamble': pream,
#})

from sabc.utils import _mmd

METRIC = "euclidean"

def metrics(files, true_samples, thin=1, final=-1, sliced=True,
			gaussian_hyperparam=None, verbose=True):

	"""
	ns and seeds are iterables containing the budgets and seeds to compute
		metrics for.

	location_template should be a string which will be formatted with {0} and
		{1} corresponding to the seed and budget, respectively.

	true_samples should be of shape (n_samples, dim)

	thin is an integer determining the number of samples skipped over to thin

	sliced is a bool. True if Wasserstein distance is to be computed in sliced
		sense
	"""

	swds, mmds, meandists = [], [], []
	true_mean = np.mean(true_samples, axis=0)
	if gaussian_hyperparam is None:
		print("Computing Gaussian kernel hyperparam...")
		#gaussian_hyperparam = _mmd._get_rbf_bandwidth(true_samples)
		gaussian_hyperparam = _mmd._med_heuristic(true_samples)
		print("Estimated as {0}".format(gaussian_hyperparam))

	if isinstance(files, str):
		load = True
		fnames = glob.glob(files)
	else:
		# Assume list of numpy arrays
		load = False
		fnames = files
	for fname in fnames:
		if load:
			if verbose:
				print()
				print(fname)
			# LOAD SAMPLES
			samples = np.loadtxt(fname)
		else:
			samples = fname

		# THIN SAMPLES
		if thin != 1:
			samples = samples[::thin]
		if final != -1:
			samples = samples[:final]
		if verbose:
			print("Sample shape: ", samples.shape)

		# COMPUTE WASSERSTEIN DISTANCE
		if sliced:
			swd = ot.sliced.sliced_wasserstein_distance(samples, true_samples, 
													n_projections=500)
		else:
			M = ot.dist(samples, true_samples, metric=METRIC)
			gamma, log = ot.emd([], [], M, log=True)
			swd = log["cost"]
		swds.append(swd)

		# COMPUTE MMD
		mmd = _mmd.compute_mmd(samples, true_samples, c=gaussian_hyperparam)
		mmds.append(mmd)

		# COMPUTE DISTANCES BETWEEN MEANS
		meandist = np.sum((np.mean(samples, axis=0) - true_mean)**2)
		meandists.append(meandist)

		# PRINT VALUES
		if verbose:
			print(swd, mmd, meandist)

	return swds, mmds, meandists, gaussian_hyperparam

def make_boxplots(labels, *args, ylabels=None, colors=None, mcolor=None,
				  ygrid=False, fontsize=12, reverse=False, xticks=None, order=None,
				  fname=False):

	plt.rcParams['font.size'] = fontsize
	if labels is None:
		labels = range(len(args[0]))
	if reverse:
		labels = labels[::-1]
		new_args = []
		for arg in args:
			new_arg = []
			for l in arg:
				new_arg = [l] + new_arg
			new_args.append(new_arg)
		colors = colors[::-1]
		args = new_args
	elif not order is None:
		new_labels = [labels[i] for i in order]
		labels = new_labels
		new_args = []
		for arg in args:
			new_arg = [arg[i] for i in order]
			new_args.append(new_arg)
		colors = [colors[i] for i in order]
		args = new_args
	fig, axes = plt.subplots(1, len(args), figsize=(15, 5))
	#fig.tight_layout()
	fig.subplots_adjust(wspace=0.1)
	abv2ylabel = {"wass":r"$\mathcal{W}_1(\hat{\pi}_{\rm ABC}, \hat{\pi}_{\cdot\mid \mathbf{y}})$",#"Wasserstein distance between posteriors", 
				  "mmd":r"${\mathrm{MMD}}^2(\hat{\pi}_{\rm ABC}, \hat{\pi}_{\cdot\mid \mathbf{y}})$",#"MMD between posteriors",
				  "md":r"$\|{\hat{\boldsymbol{\theta}}_{\rm ABC} - \hat{\boldsymbol{\theta}}_{\rm True}}\|^2$"}
	i = 0
	if not (ylabels is None):
		assert len(ylabels) == len(args), "Must provide ylabels for all plots"
		ylabels = [abv2ylabel[lab] for lab in ylabels]
	while i < len(args):
		data = args[i]
		if i > 0:
			labels = ['' for l in labels]
			axes[i].tick_params(left=False, right=False)
		if colors is None:
			bploti = axes[i].boxplot(data, labels=labels, vert=False, 
									 patch_artist=True, notch=True)
		elif not colors is None:
			bploti = axes[i].boxplot(data, labels=labels, vert=False,
									 patch_artist=True, notch=True)
			for bplot, color in zip(bploti["boxes"], colors):
				bplot.set_facecolor(color)
		if not mcolor is None:
			for median in bploti["medians"]:
				median.set_color(mcolor)
		if not xticks is None:
			axes[i].set_xticks(xticks[i])
		axes[i].set_xlabel(ylabels[i])
		axes[i].grid(ygrid)
		i += 1
	if not fname is False:
		plt.savefig(fname, dpi=500, format='png', bbox_inches='tight')
	plt.show()
	return axes
