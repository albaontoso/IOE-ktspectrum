#########################
###    IOE kernels    ###
#########################

import numpy as np
import scipy as sp
import math
import sys
import scipy.special as special
from scipy.integrate import quad
from basics import *
from bdmps import * 

class IOEkernels(Definitions):
    def __init__(self):
        super().__init__()

    def broad_integrand(self,tt,w,k,mode='broad'): # Eq.3.52 in the draft
        t = tt/self.GeVtoFm
        alphabar = self.alpha_s*self.Cr/math.pi
        prefactor = -math.pi*alphabar/pow(k,4)
        return np.real(prefactor*self.Cot(w,t,0)*self.Qs2_zero(w,t)*self.Ia(pow(k,2)/self.Phat2(w,t,0,mode),pow(k,2)/self.QC2_full(w,mode)))

    def broad(self,w,k): 
        tmin = 0
        tmax = self.L*self.GeVtoFm
        return quad(self.broad_integrand,tmin,tmax,args=(w,k))[0]/self.GeVtoFm

    def inout_integrand(self,tt,w,k): # Eq.3.65 in the draft
        t = tt/self.GeVtoFm
        alphabar = self.alpha_s*self.Cr/math.pi
        prefactor = 2*alphabar*self.qhat0*math.pi/(pow(k,4))
        sfunction = self.Ib(pow(k,2)/self.KhatInOut(w,t),pow(k,2)/(self.QC2_full(w)*pow(self.C(w,self.L,t),2)))
        return np.real(prefactor*pow(self.C(w,self.L,t),2)*sfunction*np.exp(-complex(0,1)*pow(k,2)*np.tan(self.Omega(w)*(self.L-t))/(2*w*self.Omega(w))))
    
    def inout(self,w,k): 
        tmin = 0
        tmax = self.L*self.GeVtoFm
        return quad(self.inout_integrand,tmin,tmax,args=(w,k))[0]/self.GeVtoFm

    def inin_integrand_1(self,tt1,tt2,w,k): # Eq.3.54 in the draft
        t1 = tt1/self.GeVtoFm
        t2 = tt2/self.GeVtoFm
        alphabar = self.alpha_s*self.Cr/math.pi
        prefactor = complex(0,1)*alphabar*self.qhat0*math.pi/(2*w*pow(k,4))
        term1 = np.exp(-pow(k,2)/self.Phat2(w,t2,t1))/pow(self.Rhat(w,t2,t1),2)
        term2 = self.Qs2(w,t2)*self.Ia(pow(k,2)/self.KhatInIn(w,t2,t1),pow(k,2)*pow(self.Rhat(w,t2,t1),2)/self.QC2_full(w))
        term3 = 2*self.C(w,t1,t2)*self.Rhat(w,t2,t1)*pow(k,2)*self.Ib(pow(k,2)/self.KhatInIn(w,t2,t1),pow(k,2)*pow(self.Rhat(w,t2,t1),2)/self.QC2_full(w))
        return np.real(prefactor*term1*(term2+term3))

    def inin_integrand(self,t2,w,k):
        t1min = 0
        t1max = t2
        return quad(self.inin_integrand_1,t1min,t1max,args=(t2,w,k),epsabs=1e-4,epsrel=1e-4)[0]

    def inin(self,w,k):
        t2min = 0
        t2max = self.L*self.GeVtoFm
        return quad(self.inin_integrand,t2min,t2max,args=(w,k),epsabs=1e-4,epsrel=1e-4)[0]/pow(self.GeVtoFm,2)

    def spectrum(self,w,k): # In-In + In-Out + Broad
        return self.inin(w,k) + self.inout(w,k) + self.broad(w,k)

    def ioe_integrated_spectrum_int(self,ss,w): # Eq. 2.36 in the draft (here the LO has been subtracted)
        s = ss/self.GeVtoFm
        alphabar = self.alpha_s*self.Cr/math.pi
        prefactor = alphabar*self.qhat0/2
        k2 = -complex(0,1)*w*self.Omega(w)*(1./np.tan(self.Omega(w)*s)-np.tan(self.Omega(w)*(self.L-s)))/2.
        integrand_s = (np.log(k2/self.QC2_full(w))+self.gammaE)/k2
        return prefactor*integrand_s

    def ioe_integrated_spectrum(self,w):
        smin = 0.
        smax = self.L*self.GeVtoFm
        return quad(self.ioe_integrated_spectrum_int,smin,smax,args=(w))[0]/self.GeVtoFm

    def ioe_broadening(self,k,mode='broad'): # Eq.2.19 in the draft (here the LO has been subtracted)
        x = pow(k,2)/self.QC2_full(1,mode)
        lmbda = 1./np.log(self.QC2_full(1,mode)/pow(self.mu_star,2))
        prefactor= -4*math.pi*np.exp(-x)/self.QC2_full(1,mode)
        return prefactor*lmbda*(np.exp(x)-2.+(1.-x)*sc.expi(x)-np.log(4*x))
