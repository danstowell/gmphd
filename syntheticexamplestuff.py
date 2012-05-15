#!/usr/bin/env python

# functions, classes and constants used in the synthetic GM-PHD run scripts.
# (c) 2012 Dan Stowell and Queen Mary University of London.
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

from gmphd import *
from numpy.random import rand


# The range of the space
span = (0, 60)
slopespan = (-2, 3)  # currently only used for clutter generation / inference

################################################################################
# pre-made transition/observation setups:

# state is [x, dx, offset].T, where the actual "location" in the physical sense is x+offset.
resfac = 0.95
transntypes = {
	'fixedvel': array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),  # simple fixed-velocity state update
	'vibrato': array([[1-resfac,1,0], [0-resfac,1,0], [0,0,1]])  # simple harmonic motion
	}

obsntypes = {
	# 1D spectrum-type - single freq value per bin
	'spect': {'obsnmatrix': array([[1, 0, 1]]), 
		  'noisecov': [[0.5]], 
		  'obstospec': array([[1]])
	},
	# 2D chirp-type [start, end]
	'chirp': {'obsnmatrix': array([[1, -0.5, 1], [1, 0.5, 1]]), 
		  'noisecov': [[0.5], [0.5]], 
		  'obstospec':  array([[0.5, 0.5]])
	}
	}

# Note: I have noticed that the birth gmm needs to be narrow/fine, because otherwise it can lead the pruning algo to lump foreign components together
#birthgmm = [GmphdComponent(1.0, [0, 0, centre + 0.5], [[10.0, 0, 0], [0, 0.1, 0], [0, 0, 3]]) for centre in range(5, 55, 1)]  # fine carpet
birthgmm = [GmphdComponent(1.0, [x, 0, offset], [[10.0, 0, 0], [0, 0.1, 0], [0, 0, 3]]) \
	for offset in range(5, 57, 2) for x in range(-4, 6, 2)]  # fine carpet

###############################################################
class TrackableThing:
	def __init__(self, obsnmatrix, transnmatrix):
		self.state = sampleGm(birthgmm)		
		self.state = reshape(self.state, (size(self.state), 1)) # enforce column vec
		self.alive = True
		self.obsnmatrix = obsnmatrix
		self.transnmatrix = transnmatrix
	def updateState(self):
		self.state = dot(self.transnmatrix, self.state)
	def observe(self):
		return dot(self.obsnmatrix, self.state)

############################################################################
# utility functions

def clutterintensityfromtot(clutterintensitytot, obsntype):
	"from the total clutter, calculate the point-density of it"
	if obsntype == 'spect':
		clutterrange = (span[1] - span[0])
	else:
		clutterrange = (span[1] - span[0]) * (slopespan[1] - slopespan[0])
	return float(clutterintensitytot) / float(clutterrange)

def updatetrueitems(trueitems, survivalprob, birthprob, obsnmatrix, transnmatrix):
	"update true state of ensemble - births, deaths, movements"
	for item in trueitems:
		item.updateState()
		if (rand() >= survivalprob) or (int(round(item.observe()[0])) >= span[1]) or (int(round(item.observe()[0])) < span[0]):
			item.alive = False
	trueitems = filter(lambda x: x.alive, trueitems)
	if rand() < birthprob:
		trueitems.append(TrackableThing(obsnmatrix, transnmatrix))
	print "True states:"
	for item in trueitems:
		print list(item.state.flat)
	return trueitems

def getobservations(trueitems, clutterintensitytot, obsntype, directlystatetospec, detectprob):
	"returns (observationsset, groundtruth)"
	groundtruth = [0 for _ in range(span[0], span[1])]  # simple binary spectrogram-like
	obsset      = []  # set-valued
	# clutter
	numclutter = poissonSample(clutterintensitytot)
	print "clutter generating %i items" % numclutter
	for _ in range(numclutter):
		index = int(round(span[0] + rand() * (span[1] - span[0])))
		clutterslope = int(round(slopespan[0] + rand() * (slopespan[1] - slopespan[0])))
		if obsntype == 'spect':
			obsset.append([[index]])  # spectrum-like
		else:
			obsset.append([[index-clutterslope], [index+clutterslope]])  # chirp-like
	# true
	for item in trueitems:
		bin = int(round(dot(directlystatetospec, item.state)))   # project state space directly into spec
		if bin > -1 and bin < len(groundtruth):
			groundtruth[bin] = 1
		if rand() < detectprob:
			theobservation = item.observe()
			intobs = around(theobservation).astype(int).tolist()  # round to integer, keeping array shape
			obsset.extend([intobs])
	return (obsset, groundtruth)

def updateandprune(g, obsset):
	print "-------------------------------------------------------------------"
	g.update(obsset) # here we go!
	g.prune(maxcomponents=50, mergethresh=0.15)
	print "intensity gmm offsets, after pruning:"
	print sorted([round(comp.loc[2], 1) for comp in g.gmm])

def collateresults(g, obsset, bias, obsntype, directlystatetospec, trueitems, groundtruth):
	#meh: intensity = g.gmmevalalongline([[-5,5], [0,0], [span[0]+5,span[1]-5]], span[1]-span[0])
	intensity = g.gmmevalgrid1d(span, span[1]-span[0], 2)  # "2" means just use the "offset" dimension

	integral = sum(array([comp.weight for comp in g.gmm]))

	# Get the estimated items, and also convert them back to vector representation for easy plotting
	#estitems = g.extractstates(bias=bias)
	estitems = g.extractstatesusingintegral(bias=bias)
	print "estimated %i items present" % len(estitems)
	estspec = [0 for _ in range(span[0], span[1])]
	for x in estitems:
		bin = int(round(dot(directlystatetospec, x)))   # project state space directly into spec
		if bin > -1 and bin < len(estspec):
			estspec[bin] += 1

	obsspec = obsFrameToSpecFrame(obsset, obsntype)
	return {'trueitems': trueitems, 'groundtruth': groundtruth, 'obsspec': obsspec, 'intensity': intensity, 
			'estspec': estspec, 'estitems': estitems, 'integral': integral}


def obsFrameToSpecFrame(obsset, obsntype):
	"Convert an observation frame (a SET of observation data) into a specgram-type VECTOR frame, for easiest plotting."
	obsspec = [0 for _ in range(span[0], span[1])]
	transform = obsntypes[obsntype]['obstospec']
	for anobs in obsset:
		bin = int(round(dot(transform, anobs).flat[0]))
		if bin > -1 and bin < len(obsspec):
			obsspec[bin] = 1
	return obsspec

def poissonSample(lamb):
	"Sample from a Poisson distribution. Algorithm is due to Donald Knuth - see wikipedia/Poisson_distribution"
	l = exp(-lamb)
	k = 0
	p = 1
	while True:
		k += 1
		p *= rand()
		if p <= l:
			break
	return k - 1

#def dist0(a, b):
#	"hamming distance between two equal-shape matrices, ASSUMED BINARY 1/0, returned as a fraction of the max possible"
#	a = array(a)
#	b = array(b)
#	return sum((a < 0.5) != (b < 0.5)) / float(len(a.flat))

def calcroc(gt, ob):
	"given two binary matrices, groundtruth and observed, returns (FPR, TPR), i.e. an x-y co-ordinate for a ROC plot"
	print "-=-=-=-=-=-CALCROC-=-=-=-=-=-=-"
	print gt
	print ob
	gt = array(gt) > 0
	ob = array(ob) > 0
	print gt
	print ob
	tp = sum(gt & ob)
	fn = sum(gt & (~ob))
	fp = sum((~gt) & ob)
	tn = sum((~gt) & (~ob))
	print "tp %i, fn %i, fp %i, tn %i" % (tp, fn, fp, tn)
	tpr = tp/float(tp+fn)
	fpr = fp/float(fp+tn)
	return (fpr, tpr)

