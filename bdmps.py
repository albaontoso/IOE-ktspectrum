#########################
###   BDMPS kernels  ###
########################

import numpy as np
import scipy as sp
import math
import sys
import scipy.special as special
from scipy.integrate import quad
from basics import *


class BDMPSkernels(Definitions):
    def __init__(self):
        super().__init__()

    def inin_integrand(self,tt,w,k,mode='broad'): # Eq. 3.27 in the draft
        t = tt/self.GeVtoFm
        prefactor = 8.*self.alpha_s*self.Cr # simplified the pi factors
        Qs2 = self.qhat(w,mode)*(self.L-t)
        den = (Qs2-complex(0,1)*2.*w*self.Omega(w)*self.cot(self.Omega(w)*t)) 
        return np.real(prefactor*self.Omega(w)*self.cot(self.Omega(w)*t)*np.exp(-pow(k,2)/den)/den)

    def inin(self,w,k): 
        tmin = 0;
        tmax = self.L*self.GeVtoFm
        return quad(self.inin_integrand,tmin,tmax,args=(w,k))[0]/self.GeVtoFm

    def inout(self,w,k): # Eq. 3.32 in the draft
        alphabar = self.alpha_s*self.Cr/math.pi
        prefactor = 8*alphabar*math.pi/pow(k,2)
        arg = complex(0,1)*pow(k,2)/(2*w*self.Omega(w)*self.cot(self.Omega(w)*self.L))
        return prefactor*np.real(np.exp(-arg)-1.0)

    def spectrum(self,w,k): # In-In + In-Out
        return self.inin(w,k)+self.inout(w,k)    

    def integrated_spectrum(self,w): # Eq. 2.31 in the draft
        alphabar = self.alpha_s*self.Cr/math.pi
        return np.real(2*alphabar*np.log(np.abs(np.cos(self.Omega(w)*self.L))))

    def broadening(self,k,mode='broad'): # First term in Eq.2.19 in the draft
        x = pow(k,2)/self.QC2_full(1,mode)
        return 4*math.pi*np.exp(-x)/self.QC2_full(1,mode)




