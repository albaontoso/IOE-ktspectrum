############################
###   Example of usage  ####
############################

import numpy as np
import math
from basics import *
from bdmps import * 
from ioe import * 
from glv import *

bdmps = BDMPSkernels()
ioe = IOEkernels()
glv = GLVkernels()
definition = Definitions()

# Values of omega and kt

kt_scale = np.sqrt(definition.qhat0*definition.L)
w_scale = definition.wc

w = 0.5*w_scale;
k = 2*kt_scale;

# Spectrum results in GeV^-2
print ('(2pi)^wdI/dwd^2k [GeV^-2]')
print(' LO = ', bdmps.spectrum(w,k), ' NLO = ', ioe.spectrum(w,k), ' LO+NLO = ', bdmps.spectrum(w,k)+ioe.spectrum(w,k), 'GLV = ', glv.spectrum(w,k) )
