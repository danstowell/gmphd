#!/usr/bin/env python

# script to generate ROC plot from multiple GM-PHD runs with different bias levels
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from syntheticexamplestuff import *

###############################################################
# user config options:
niters = 100
nruns = 20
birthprob = 0.1  # 0.05 # 0 # 0.2
survivalprob = 0.975 # 0.95 # 1
detectprob =  0.95#  0.999
clutterintensitytot = 5 #2 #4   # typical num clutter items per frame
biases = [1, 2, 4, 8, 16]   # tendency to prefer false-positives over false-negatives in the filtered output
obsntypenames = ['chirp', 'spect']
transntype = 'vibrato' # 'fixedvel' or 'vibrato'

###############################################################
# setting up variables
transnmatrix = transntypes[transntype]

birthintensity1 = birthprob / len(birthgmm)
print "birthgmm: each component has weight %g" % birthintensity1
for comp in birthgmm:
	comp.weight = birthintensity1

rocpoints = {}
for obsntype in obsntypenames:
	obsnmatrix = obsntypes[obsntype]['obsnmatrix']
	directlystatetospec = dot(obsntypes[obsntype]['obstospec'], obsnmatrix)
	clutterintensity = clutterintensityfromtot(clutterintensitytot, obsntype)
	print "clutterintensity: %g" % clutterintensity
	rocpoints[obsntype] = [(0,0)]

	# NOTE: all the runs are appended into one long "results" array! Can calc the roc point in one fell swoop, no need to hold separate.
	# So, we concatenate one separate resultlist for each bias type.
	# Then when we come to the end we calculate a rocpoint from each resultlist.
	results = { bias: [] for bias in biases }

	for whichrun in range(nruns):
		print "===============================obsntype %s, run %i==============================" % (obsntype, whichrun)
		### Initialise the true state and the model:
		trueitems = []
		g = Gmphd(birthgmm, survivalprob, 0.7, transnmatrix, 1e-9 * array([[1,0,0], [0,1,0], [0,0,1]]), 
				obsnmatrix, obsntypes[obsntype]['noisecov'], clutterintensity)

		for whichiter in range(niters):
			print "--%i----------------------------------------------------------------------" % whichiter
			# the "real" state evolves
			trueitems = updatetrueitems(trueitems, survivalprob, birthprob, obsnmatrix, transnmatrix)
			# we make our observations of it
			(obsset, groundtruth) = getobservations(trueitems, clutterintensitytot, obsntype, directlystatetospec, detectprob)
			print "OBSSET sent to g.update():"
			print obsset
			# we run our inference using the observations
			updateandprune(g, obsset)

			for bias in biases:
				resultdict = collateresults(g, obsset, bias, obsntype, directlystatetospec, trueitems, groundtruth)
				results[bias].append(resultdict)

	for bias in biases:
		gt = [moment['groundtruth'] for moment in results[bias]]
		ob = [moment['estspec']     for moment in results[bias]]
		rocpoints[obsntype].append(calcroc(gt, ob))
		print "rocpoints"
		print rocpoints

	rocpoints[obsntype].append((1,1))

###############################################################
# plot the results
fig = plt.figure()

plt.hold(True)
plt.plot([p[0] for p in rocpoints['spect']], [p[1] for p in rocpoints['spect']], 'r+--', label='spect')
plt.plot([p[0] for p in rocpoints['chirp']], [p[1] for p in rocpoints['chirp']], 'b*-', label='chirp')
plt.legend(loc=4)
plt.title('GMPHD, synthetic data (%s, %i runs per point, avg clutter %g)' % (transntype, nruns, clutterintensitytot))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(xmin=0, xmax=0.4)
plt.ylim(ymin=0, ymax=1)


plt.savefig("plot_synthroc.pdf", papertype='A4', format='pdf')
fig.show()
raw_input("Press Enter to continue...")

