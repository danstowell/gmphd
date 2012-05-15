				   ======================================
								   gmphd
				   GM-PHD filter implementation in python
							  by Dan Stowell
				   ======================================

This is a Python implementation of the Gaussian mixture PHD filter
(probability hypothesis density filter) described in:

B. N. Vo and W. K. Ma. The gaussian mixture probability hypothesis density filter. 
   IEEE Transactions on Signal Processing, 54(11):4091Ð4104, 2006.
   DOI: 10.1109/TSP.2006.881190

It requires Numpy, and the demo scripts require matplotlib.
Tested with Python 2.7.


DIFFERENCES
===========

There are some differences from the GM-PHD algorithm described in the paper:

* I have not implemented "spawning" of new targets from old ones, since I don't 
  need it. It would be straightforward to add it - see the original paper.

* Weights are adjusted at the end of pruning, so that pruning doesn't affect
  the total weight allocation.

* I provide an alternative approach to state-extraction (an alternative to
  Table 3 in the original paper) which makes use of the integral to decide how
  many states to extract.


USAGE
=====

The file "syntheticexample.py" is a python script which runs the filter over a
synthetic randomly-generated scene, in which objects have 3D state and generate
chirp-like observations. I suggest you start by looking at that script. But
for a quick look at the API here's a very simple bit of python:

from gmphd import *
g = Gmphd([GmphdComponent(1, [100], [[10]])], 0.9, 0.9, [[1]], [[1]], [[1]], [[1]], 0.000002)
g.update([[30], [67.5]])
g.gmmplot1d()
g.prune()
g.gmmplot1d()


LICENCE
=======

(c) 2012 Dan Stowell and Queen Mary University of London.
All rights reserved.

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
