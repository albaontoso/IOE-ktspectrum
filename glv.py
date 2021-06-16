#########################
###   GLV kernels   ####
########################

import numpy as np
import scipy as sp
import math
import sys
import scipy.special as special
from scipy.integrate import quad
from basics import *
from scipy.special import kn
from scipy.special import jv

class GLVkernels(Definitions):
    def __init__(self):
        super().__init__()

    def glv_integrand(self,x,w,k): # Eq.3.92 in the draft
        alphabar = self.alpha_s*self.Cr/math.pi
        prefactor = 2*alphabar*self.qhat0*pow(self.L,3)*math.pi/pow(w,2)
        u = self.L*pow(k,2)/(2*w)
        gamma = pow(self.mu_gw,2)*self.L/(2*w)
        term1 = (x-np.sin(x))/pow(x,2)
        num = gamma+u-x
        den = pow(pow(u,2)+2*u*(gamma-x)+pow(gamma+x,2),1.5)
        return prefactor*term1*num/den
    
    def spectrum(self,w,k):
        xmin = 0
        xmax = np.inf
        return quad(self.glv_integrand,xmin,xmax,args=(w,k))[0]

    def glv_integrated_spectrum_int(self,u,w): # Eq.3.15 in 1903.00506 
        mu2_gw = pow(self.mu_gw,2)
        y = mu2_gw*self.L/(2*w)
        alphabar = self.alpha_s*self.Cr/math.pi
        factor = alphabar*self.qhat0*pow(self.L,2)/w
        return factor*(u-np.sin(u))/(pow(u,2)*(u+y))

    def glv_integrated(self,w):
        umin = 0.
        umax = np.inf
        return quad(self.glv_integrated_spectrum_int,umin,umax,args=(w))[0] 
    
    def broadening_int(self,x,k): # Eq.2.2 in the draft
        mu2_gw = pow(self.mu_gw,2)
        full_potential = np.sqrt(mu2_gw)*x*kn(1,np.sqrt(mu2_gw)*x)
        return np.exp(-self.Qs2zero/mu2_gw)*x*jv(0,k*x)*(np.exp(self.Qs2zero*full_potential/mu2_gw)-1.)
    
    def broadening(self,k):
        return 2*math.pi*quad(self.broadening_int,0,np.inf,args=(k))[0]