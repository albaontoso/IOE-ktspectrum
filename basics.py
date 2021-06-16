#########################################
## Medium constants and some functions ##
#########################################

import numpy as np
import scipy as sp
import math
import sys
import scipy.special as sc

class Definitions:
        def __init__(self):
            # Medium parameters
            self.GeVtoFm = 0.2; # conversion factor
            self.Cr = 4./3; # color factor
            self.alpha_s = 0.28; # fixed coupling
            self.mu_star = 0.3545; # infra-red regulator in GeV
            self.gammaE = 0.577 # Euler-Mascheroni constant
            self.mu_gw =np.sqrt(4*pow(self.mu_star,2)/np.exp(-1+2*self.gammaE)) # infra-red regulator for GW 
            self.L = 6./self.GeVtoFm # length of the medium in GeV
            self.qhat0 = 0.1557345
            self.Qs2zero = self.qhat0*self.L
            self.wc = self.qhat0*pow(self.L,2)

        def QC2_full(self,w,mode='rad'): # Solving the trascendental Eq.2.42 for the matching scale
            maxit = 50; # number of iterations
            tol = 1e-2; # tolerance
            mu2 = pow(self.mu_star,2) 
            xnext = 0.
            if mode=='rad': # See Eq. 2.41 in the draft
                prefactor = np.sqrt(w*self.qhat0)
                xcur = self.Qs2zero
                for i in range(0,maxit):
                    if(xcur/mu2<1.0):
                        print("Warning: Q < mu_star, setting Qc=0")
                        return 0
                    else:
                        xnext=prefactor*np.sqrt(np.log(xcur/mu2));
                        error=abs(xnext-xcur)/xcur;
                        if error<tol:
                            break;
                        xcur=xnext;
                return xnext;
            if mode=='broad': # See Eq. 2.18 in the draft
                xcur=self.Qs2zero;
                xnext=0.;
                for i in range(0,maxit):
                    if(xcur/mu2<1.0):
                        print("Warning: Q < mu_star, setting Qc=0")
                        return 0
                    else:
                        xnext=self.Qs2zero*np.log(xcur/mu2);
                        error=abs(xnext-xcur)/xcur;
                        if error<tol:
                            break;
                        xcur=xnext;
                return xnext
        
        def qhat(self,w,mode='rad'): 
            Qc2 = self.QC2_full(w,mode)
            if(Qc2/pow(self.mu_star,2)<1.0):
                print("Warning: Q < mu_star, setting qhat=0")
                return 0
            else:
                return self.qhat0*np.log(Qc2/pow(self.mu_star,2))
        
        def Omega(self,w,mode='rad'): # Eq.3.20 in the draft 
            return complex(1,-1)*np.sqrt(self.qhat(w,mode)/w)/2 

        def Qs2_zero(self,w,t): # Eq.2.17 in the draft
            return self.qhat0*(self.L-t)

        def Qs2(self,w,t,mode='broad'): # Eq. 3.79 in the draft
            return self.qhat(w,mode)*(self.L-t)

        def cot(self,x): # Somehow it is not defined in numpy
            return 1./np.tan(x)

        def Cot(self,w,x,y,mode='rad'): # below Eq.3.25 in the draft
            return self.Omega(w,mode)*self.cot(self.Omega(w,mode)*(x-y))
        
        def Ia(self,x,y): # Eq.3.45 in the draft
            return 8*pow(x,2)*np.exp(-x)*(-2+np.exp(x))+8*pow(x,2)*np.exp(-x)*(1-x)*(sc.expi(x)-np.log(4*pow(x,2)/y))

        def Ib(self,x,y): # Eq.3.46 in the draft 
            return -4*x*(1.-np.exp(-x))+4*pow(x,2)*np.exp(-x)*(sc.expi(x)-np.log(4*pow(x,2)/y))

        def C(self,w,x,y): # Eq.3.25 in the draft
            return np.cos(self.Omega(w)*(x-y))

        def S(self,w,x,y): # Eq.3.25 in the draft
            return np.sin(self.Omega(w)*(x-y))/self.Omega(w)

        def KhatInOut(self,w,x):# Eq.3.83 in the draft
            return 2.*complex(0,1)*w*self.Omega(w)*pow(self.C(w,self.L,x),2)*(np.tan(self.Omega(w)*(self.L-x))-self.cot(self.Omega(w)*x))

        def Phat2(self,w,x,y,mode='rad'): # Eq.3.81 in the draft
            return self.Qs2(w,x)-2*complex(0,1)*w*self.Cot(w,x,y,mode)

        def Rhat(self,w,x,y): # Eq.3.81 in the draft (third line)
            return -2*complex(0,1)*w/(self.Phat2(w,x,y)*self.S(w,x,y))
        
        def KhatInIn(self,w,x,y): # Eq. 3.81 in the draft (second line)
            return complex(0,1)*pow(self.S(w,x,y)*self.Phat2(w,x,y),2)*(-self.Cot(w,y,x)+self.Cot(w,y,0)+2*complex(0,1)*w/(pow(self.S(w,x,y),2)*self.Phat2(w,x,y)))/(2*w)
