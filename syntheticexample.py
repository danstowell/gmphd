#!/usr/bin/env python

# script which runs a synthetic model plus clutter and GM-PHD tracking,
#  and generates a plot of how things evolved over time.
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
birthprob = 0.1  # 0.05 # 0 # 0.2
survivalprob = 0.975 # 0.95 # 1
detectprob =  0.95#  0.999
clutterintensitytot = 5 #2 #4   # typical num clutter items per frame
bias = 2 #8   # tendency to prefer false-positives over false-negatives in the filtered output
initcount = 0 #2  # 4
obsntype = 'chirp' # 'chirp' or 'spect'
transntype = 'vibrato' # 'fixedvel' or 'vibrato'

###############################################################
# setting up variables
transnmatrix = transntypes[transntype]
obsnmatrix = obsntypes[obsntype]['obsnmatrix']
directlystatetospec = dot(obsntypes[obsntype]['obstospec'], obsnmatrix)
#### This one deliberately ignores the vibrato component - doing it for the plot in the paper
directlystatetospec001 = array([[0., 0., 1.]])

birthintensity1 = birthprob / len(birthgmm)
print("birthgmm: each component has weight %g" % birthintensity1)
for comp in birthgmm:
	comp.weight = birthintensity1

clutterintensity = clutterintensityfromtot(clutterintensitytot, obsntype)
print("clutterintensity: %g" % clutterintensity)

### Create the "true" state and the model:
trueitems = [TrackableThing(obsnmatrix, transnmatrix) for _ in range(initcount)]
if initcount != 0:
	print("True states at init:")
	for item in trueitems:
		print(list(item.state.T))

g = Gmphd(birthgmm, survivalprob, 0.7, transnmatrix, 1e-9 * array([[1,0,0], [0,1,0], [0,0,1]]), obsnmatrix, obsntypes[obsntype]['noisecov'], clutterintensity)

###############################################################
results = []
for whichiter in range(niters):
	print("--%i----------------------------------------------------------------------" % whichiter)
	# the "real" state evolves
	trueitems = updatetrueitems(trueitems, survivalprob, birthprob, obsnmatrix, transnmatrix)
	# we make our observations of it
	(obsset, groundtruth) = getobservations(trueitems, clutterintensitytot, obsntype, directlystatetospec, detectprob)
	print("OBSSET sent to g.update():")
	print(obsset)
	# we run our inference using the observations
	updateandprune(g, obsset)
	resultdict = collateresults(g, obsset, bias, obsntype, directlystatetospec, trueitems, groundtruth)

	# also manually grab a version using the 001 matrix, for the paper:
	resultdict001 = collateresults(g, obsset, bias, obsntype, directlystatetospec001, trueitems, groundtruth)
	resultdict['estspec001']     = resultdict001['estspec']

	results.append(resultdict)

###############################################################
# plot the results
fig = plt.figure()

#dist_obs   = dist0(array([moment['groundtruth'] for moment in results]), \
#                   array([moment['obsspec']     for moment in results]))
#dist_infer = dist0(array([moment['groundtruth'] for moment in results]), \
#                   array([moment['estspec']     for moment in results]))

# True trajectories:
ax = fig.add_subplot(511)
ax.imshow(array([moment['groundtruth'] for moment in results]).T, aspect='auto', interpolation='nearest', cmap=cm.binary)    #, norm=normer, cmap=cmap)
plt.ylabel('True', fontsize='x-small')
plt.xticks( fontsize='x-small' )
plt.yticks( arange(0, 60, 10), ('', '', '', '', '', '') )

# Noisy observations:
ax = fig.add_subplot(512)
ax.imshow(array([moment['obsspec'] for moment in results]).T, aspect='auto', interpolation='nearest', cmap=cm.binary)    #, norm=normer, cmap=cmap)
plt.ylabel('Observed', fontsize='x-small')
plt.xticks( fontsize='x-small' )
plt.yticks( arange(0, 60, 10), ('', '', '', '', '', '') )

## Intensity plot (GMM density):
#ax = fig.add_subplot(521)
#ax.imshow((array([moment['intensity'] for moment in results]).T), aspect='auto')    #, norm=normer, cmap=cmap)
#plt.ylabel('Intensity\n(%g)' % (max([max(moment['intensity']) for moment in results])))
plt.xticks( fontsize='x-small' )
plt.yticks( arange(0, 60, 10), ('', '', '', '', '', '') )

# Estimated locations:
ax = fig.add_subplot(513)
ax.imshow(array([[min(x,1.0) for x in moment['estspec']] for moment in results]).T, aspect='auto', interpolation='nearest', cmap=cm.binary)
plt.ylabel('Estimated', fontsize='x-small')
plt.xticks( fontsize='x-small' )
plt.yticks( arange(0, 60, 10), ('', '', '', '', '', '') )
ax = fig.add_subplot(514)
ax.imshow(array([[min(x,1.0) for x in moment['estspec001']] for moment in results]).T, aspect='auto', interpolation='nearest', cmap=cm.binary)
plt.ylabel('Estimated\n(no vibrato)', fontsize='x-small')
plt.xticks( fontsize='x-small' )
plt.yticks( arange(0, 60, 10), ('', '', '', '', '', '') )

# True and est count:
ax = fig.add_subplot(515)
ax.plot(range(len(results)), [sum(array(moment['groundtruth'])) for moment in results], 'b-', \
        range(len(results)), [moment['integral'] for moment in results], 'r-')
plt.ylim(ymin=0)
plt.ylabel('True&Est #', fontsize='x-small')
plt.xticks( fontsize='x-small' )
plt.yticks( fontsize='x-small' )

plt.savefig("plot_testgmphd.pdf", papertype='A4', format='pdf')
fig.show()
#raw_input("Press Enter to continue...")

