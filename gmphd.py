#!/usr/bin/env python

# GM-PHD implementation in python by Dan Stowell.
# Based on the description in Vo and Ma (2006).
# (c) 2012 Dan Stowell and Queen Mary University of London.
# All rights reserved.
#
# NOTE: I AM NOT IMPLEMENTING SPAWNING, since I don't need it.
#   It would be straightforward to add it - see the original paper for how-to.
"""
This file is part of gmphd, GM-PHD filter in python by Dan Stowell.

    gmphd is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gmphd is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gmphd.  If not, see <http://www.gnu.org/licenses/>.
"""

simplesum = sum   # we want to be able to use "pure" sum not numpy (shoulda namespaced)
from numpy import *
import numpy.linalg
from copy import deepcopy
from operator import attrgetter

myfloat = float64

class GmphdComponent:
	"""Represents a single Gaussian component, 
	with a float weight, vector location, matrix covariance.
	Note that we don't require a GM to sum to 1, since not always about proby densities."""
	def __init__(self, weight, loc, cov):
		self.weight = myfloat(weight)
		self.loc    = array(loc, dtype=myfloat, ndmin=2)
		self.cov    = array(cov, dtype=myfloat, ndmin=2)
		self.loc    = reshape(self.loc, (size(self.loc), 1)) # enforce column vec
		self.cov    = reshape(self.cov, (size(self.loc), size(self.loc))) # ensure shape matches loc shape
		# precalculated values for evaluating gaussian:
		k = len(self.loc)
		self.dmv_part1 = (2.0 * pi) ** (-k * 0.5)
		self.dmv_part2 = power(numpy.linalg.det(self.cov), -0.5)
		self.invcov = numpy.linalg.inv(self.cov)

	def dmvnorm(self, x):
		"""Evaluate this multivariate normal component, at a location x.
		NB this does NOT APPLY THE WEIGHTING, simply for API similarity to the other method with this name."""
		x = array(x, dtype=myfloat)
		dev = x - self.loc
		part3 = exp(-0.5 * dot(dot(dev.T, self.invcov), dev))
		return self.dmv_part1 * self.dmv_part2 * part3

# We don't always have a GmphdComponent object so:
def dmvnorm(loc, cov, x):
	"Evaluate a multivariate normal, given a location (vector) and covariance (matrix) and a position x (vector) at which to evaluate"
	loc = array(loc, dtype=myfloat)
	cov = array(cov, dtype=myfloat)
	x = array(x, dtype=myfloat)
	k = len(loc)
	part1 = (2.0 * pi) ** (-k * 0.5)
	part2 = power(numpy.linalg.det(cov), -0.5)
	dev = x - loc
	part3 = exp(-0.5 * dot(dot(dev.T, numpy.linalg.inv(cov)), dev))
	return part1 * part2 * part3

def sampleGm(complist):
	"Given a list of GmphdComponents, randomly samples a value from the density they represent"
	weights = array([x.weight for x in complist])
	weights = weights / simplesum(weights)   # Weights aren't externally forced to sum to one
	choice = random.random()
	cumulative = 0.0
	for i,w in enumerate(weights):
		cumulative += w
		if choice <= cumulative:
			# Now we sample from the chosen component and return a value
			comp = complist[i]
			return random.multivariate_normal(comp.loc.flat, comp.cov)
	raise RuntimeError("sampleGm terminated without choosing a component")

################################################################################
class Gmphd:
	"""Represents a set of modelling parameters and the latest frame's
	   GMM estimate, for a GM-PHD model without spawning.
	   
	   Typical usage would be, for each frame of input data, to run:
	      g.update(obs)
	      g.prune()
	      estimate = g.extractstates()
	   
	  'gmm' is an array of GmphdComponent items which makes up
		   the latest GMM, and updated by the update() call. 
		   It is initialised as empty.

	Test code example (1D data, with new trails expected at around 100):
from gmphd import *
g = Gmphd([GmphdComponent(1, [100], [[10]])], 0.9, 0.9, [[1]], [[1]], [[1]], [[1]], 0.000002)
g.update([[30], [67.5]])
g.gmmplot1d()
g.prune()
g.gmmplot1d()

g.gmm

[(float(comp.loc), comp.weight) for comp in g.gmm]
	"""
	
	def __init__(self, birthgmm, survival, detection, f, q, h, r, clutter):
		"""
		  'birthgmm' is an array of GmphdComponent items which makes up
			   the GMM of birth probabilities.
		  'survival' is survival probability.
		  'detection' is detection probability.
		  'f' is state transition matrix F.
		  'q' is the process noise covariance Q.
		  'h' is the observation matrix H.
		  'r' is the observation noise covariance R.
		  'clutter' is the clutter intensity.
		  """
		self.gmm = []  # empty - things will need to be born before we observe them
		self.birthgmm = birthgmm
		self.survival = myfloat(survival)        # p_{s,k}(x) in paper
		self.detection = myfloat(detection)      # p_{d,k}(x) in paper
		self.f = array(f, dtype=myfloat)   # state transition matrix      (F_k-1 in paper)
		self.q = array(q, dtype=myfloat)   # process noise covariance     (Q_k-1 in paper)
		self.h = array(h, dtype=myfloat)   # observation matrix           (H_k in paper)
		self.r = array(r, dtype=myfloat)   # observation noise covariance (R_k in paper)
		self.clutter = myfloat(clutter)   # clutter intensity (KAU in paper)

	def update(self, obs):
		"""Run a single GM-PHD step given a new frame of observations.
		  'obs' is an array (a set) of this frame's observations.
		  Based on Table 1 from Vo and Ma paper."""
		#######################################
		# Step 1 - prediction for birth targets
		born = [deepcopy(comp) for comp in self.birthgmm]
		# The original paper would do a spawning iteration as part of Step 1.
		spawned = []    # not implemented
		
		#######################################
		# Step 2 - prediction for existing targets
		updated = [GmphdComponent(                        \
					self.survival * comp.weight,          \
					dot(self.f, comp.loc),                \
					self.q + dot(dot(self.f, comp.cov), self.f.T)   \
			) for comp in self.gmm]
	
		predicted = born + spawned + updated
		
		#######################################
		# Step 3 - construction of PHD update components
		# These two are the mean and covariance of the expected observation
		nu = [dot(self.h, comp.loc)                         for comp in predicted]
		s  = [self.r + dot(dot(self.h, comp.cov), self.h.T) for comp in predicted]
		# Not sure about any physical interpretation of these two...
		k = [dot(dot(comp.cov, self.h.T), linalg.inv(s[index]))
						for index, comp in enumerate(predicted)]
		pkk = [dot(eye(len(k[index])) - dot(k[index], self.h), comp.cov)
						for index, comp in enumerate(predicted)]

		#######################################
		# Step 4 - update using observations
		# The 'predicted' components are kept, with a decay
		newgmm = [GmphdComponent(comp.weight * (1.0 - self.detection), comp.loc, comp.cov) for comp in predicted]

		# then more components are added caused by each obsn's interaction with existing component
		for anobs in obs:
			anobs = array(anobs)
			newgmmpartial = []
			for j, comp in enumerate(predicted):
				newgmmpartial.append(GmphdComponent(                           \
							self.detection * comp.weight          \
								* dmvnorm(nu[j], s[j], anobs), \
							comp.loc + dot(k[j], anobs - nu[j]),   \
							comp.cov                               \
						))
	
			# The Kappa thing (clutter and reweight)
			weightsum = simplesum(newcomp.weight for newcomp in newgmmpartial)
			reweighter = 1.0 / (self.clutter + weightsum)
			for newcomp in newgmmpartial:
				newcomp.weight *= reweighter

			newgmm.extend(newgmmpartial)
	
		self.gmm = newgmm

	def prune(self, truncthresh=1e-6, mergethresh=0.01, maxcomponents=100):
		"""Prune the GMM. Alters model state.
		  Based on Table 2 from Vo and Ma paper."""
		# Truncation is easy
		weightsums = [simplesum(comp.weight for comp in self.gmm)]   # diagnostic
		sourcegmm = [comp for comp in self.gmm if comp.weight > truncthresh]
		weightsums.append(simplesum(comp.weight for comp in sourcegmm))
		origlen  = len(self.gmm)
		trunclen = len(sourcegmm)
		# Iterate to build the new GMM
		newgmm = []
		while len(sourcegmm) > 0:
			# find weightiest old component and pull it out
			windex = argmax(comp.weight for comp in sourcegmm)
			weightiest = sourcegmm[windex]
			sourcegmm = sourcegmm[:windex] + sourcegmm[windex+1:]
			# find all nearby ones and pull them out
			distances = [float(dot(dot((comp.loc - weightiest.loc).T, comp.invcov), comp.loc - weightiest.loc)) for comp in sourcegmm]
			dosubsume = array([dist <= mergethresh for dist in distances])
			subsumed = [weightiest]
			if any(dosubsume):
				#print "Subsuming the following locations into weightest with loc %s and weight %g (cov %s):" \
				#	% (','.join([str(x) for x in weightiest.loc.flat]), weightiest.weight, ','.join([str(x) for x in weightiest.cov.flat]))
				#print list([comp.loc[0][0] for comp in list(array(sourcegmm)[ dosubsume]) ])
				subsumed.extend( list(array(sourcegmm)[ dosubsume]) )
				sourcegmm = list(array(sourcegmm)[~dosubsume])
			# create unified new component from subsumed ones
			aggweight = simplesum(comp.weight for comp in subsumed)
			newcomp = GmphdComponent( \
				aggweight,
				sum(array([comp.weight * comp.loc for comp in subsumed]), 0) / aggweight,
				sum(array([comp.weight * (comp.cov + (weightiest.loc - comp.loc) \
							* (weightiest.loc - comp.loc).T) for comp in subsumed]), 0) / aggweight
					)
			newgmm.append(newcomp)

		# Now ensure the number of components is within the limit, keeping the weightiest
		newgmm.sort(key=attrgetter('weight'))
		newgmm.reverse()
		self.gmm = newgmm[:maxcomponents]
		weightsums.append(simplesum(comp.weight for comp in newgmm))
		weightsums.append(simplesum(comp.weight for comp in self.gmm))
		print("prune(): %i -> %i -> %i -> %i" % (origlen, trunclen, len(newgmm), len(self.gmm)))
		print("prune(): weightsums %g -> %g -> %g -> %g" % (weightsums[0], weightsums[1], weightsums[2], weightsums[3]))
		# pruning should not alter the total weightsum (which relates to total num items) - so we renormalise
		weightnorm = weightsums[0] / weightsums[3]
		for comp in self.gmm:
			comp.weight *= weightnorm

	def extractstates(self, bias=1.0):
		"""Extract the multiple-target states from the GMM.
		  Returns a list of target states; doesn't alter model state.
		  Based on Table 3 from Vo and Ma paper.
		  I added the 'bias' factor, by analogy with the other method below."""
		items = []
		print("weights:")
		print([round(comp.weight, 7) for comp in self.gmm])
		for comp in self.gmm:
			val = comp.weight * float(bias)
			if val > 0.5:
				for _ in range(int(round(val))):
					items.append(deepcopy(comp.loc))
		for x in items: print(x.T)
		return items

	def extractstatesusingintegral(self, bias=1.0):
		"""Extract states based on the expected number of states from the integral of the intensity.
		This is NOT in the GMPHD paper; added by Dan.
		"bias" is a multiplier for the est number of items.
		"""
		numtoadd = int(round(float(bias) * simplesum(comp.weight for comp in self.gmm)))
		print("bias is %g, numtoadd is %i" % (bias, numtoadd))
		items = []
		# A temporary list of peaks which will gradually be decimated as we steal from its highest peaks
		peaks = [{'loc':comp.loc, 'weight':comp.weight} for comp in self.gmm]
		while numtoadd > 0:
			windex = 0
			wsize = 0
			for which, peak in enumerate(peaks):
				if peak['weight'] > wsize:
					windex = which
					wsize = peak['weight']
			# add the winner
			items.append(deepcopy(peaks[windex]['loc']))
			peaks[windex]['weight'] -= 1.0
			numtoadd -= 1
		for x in items: print(x.T)
		return items

	########################################################################################
	def gmmeval(self, points, onlydims=None):
		"""Evaluates the GMM at a supplied list of points (full dimensionality). 
		'onlydims' if not nil, marginalises out (well, ignores) the nonlisted dims. All dims must still be listed in the points, so put zeroes in."""
		return [ \
			simplesum(comp.weight * comp.dmvnorm(p) for comp in self.gmm) \
				for p in points]
	def gmmeval1d(self, points, whichdim=0):
		"Evaluates the GMM at a supplied list of points (1D only)"
		return [ \
			simplesum(comp.weight * dmvnorm([comp.loc[whichdim]], [[comp.cov[whichdim][whichdim]]], p) for comp in self.gmm) \
				for p in points]

	def gmmevalgrid1d(self, span=None, gridsize=200, whichdim=0):
		"Evaluates the GMM on a uniformly-space grid of points (1D only)"
		if span==None:
			locs = array([comp.loc[whichdim] for comp in self.gmm])
			span = (min(locs), max(locs))
		grid = (arange(gridsize, dtype=float) / (gridsize-1)) * (span[1] - span[0]) + span[0]
		return self.gmmeval1d(grid, whichdim)


	def gmmevalalongline(self, span=None, gridsize=200, onlydims=None):
		"""Evaluates the GMM on a uniformly-spaced line of points (i.e. a 1D line, though can be angled).
		'span' must be a list of (min, max) for each dimension, over which the line will iterate.
		'onlydims' if not nil, marginalises out (well, ignores) the nonlisted dims. All dims must still be listed in the spans, so put zeroes in."""
		if span==None:
			locs = array([comp.loc for comp in self.gmm]).T   # note transpose - locs not a list of locations but a list of dimensions
			span = array([ map(min,locs), map(max,locs) ]).T   # note transpose - span is an array of (min, max) for each dim
		else:
			span = array(span)
		steps = (arange(gridsize, dtype=float) / (gridsize-1))
		grid = array(map(lambda aspan: steps * (aspan[1] - aspan[0]) + aspan[0], span)).T  # transpose back to list of state-space points
		return self.gmmeval(grid, onlydims)

	def gmmplot1d(self, gridsize=200, span=None, obsnmatrix=None):
		"Plots the GMM. Only works for 1D model."
		import matplotlib.pyplot as plt
		vals = self.gmmevalgrid1d(span, gridsize, obsnmatrix)
		fig = plt.figure()
		plt.plot(grid, vals, '-')
		fig.show()
		return fig
