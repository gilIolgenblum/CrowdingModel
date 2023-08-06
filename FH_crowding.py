import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize

class var:
    '''
    The basic class, contains basic class variables and methods
    '''
    R = 8.314 # J/mol
    T = 298 # K
    Vs = 0.018 # solvent molar vol in L/mol

    def cal_phiC(self):
        ''' calculates the cosolute volume fraction '''
        return np.arange(self.phiC_min, self.phiC_max, self.dphiC)
        
    def cal_osm(self):
        ''' calculates the cosolute osmotic pressure'''
        return -(np.log(1-self.phiC) + self.phiC*(1-(1/self.nu)) + self.chi*(self.phiC)**2) / self.Vs

    
    def cal_muC(self):
        ''' Calculate the chemical potential of the cosolute '''
        return (1-self.phiC)*(1-(self.nu)) + np.log(self.phiC) + self.chi*(self.nu)*((1-self.phiC)**2)

    def cal_muS(self):
        ''' Calculate the chemical potential of the solvent '''
        return np.log(1-self.phiC) + self.phiC*(1-(1/self.nu)) + self.chi*((self.phiC)**2)

    def cal_FH_free_energy(self):
        ''' Calculate the cosolute-solvent FH mixing free energy '''
        return self.phiS*np.log(self.phiS)+1/self.nu*self.phiC*np.log(self.phiC)+self.chi*self.phiS*self.phiC

    def cal_FH_entropy(self):
        ''' Calculate the cosolute-solvent FH mixing entropy '''
        return -self.phiS*np.log(self.phiS)-1/self.nu*self.phiC*np.log(self.phiC)+self.chiTS*self.phiS*self.phiC
    
    def cal_FH_enthalpy(self):
        ''' Calculate the cosolute-solvent FH mixing enthalpy '''
        return self.chiH*self.phiS*self.phiC

class Cosol(var):
    '''
    Cosolute class, contains class variable and methods that depend on the cosolute propeties.

    Args:
        nu: FH excluded volume parameter
        chi: FH non-ideal interaction parameter
        chiTs: The entropic contribution of chi
        phiC_min: Minimal concetration (volume fraction)
        phiC_max: Maximal concetration (volume fraction)
        dphiC: Concetraion step (volume fraction)
    '''
    
    def __init__(self, nu, chi, chiTS, phiC_min=0.0001, phiC_max=0.35, dphiC=0.0001):
        self.nu, self.chi, self.chiTS, self.phiC_min, self.phiC_max, self.dphiC = nu, chi, chiTS, phiC_min, phiC_max, dphiC
        self.chiH = self.chi + self.chiTS
        self.phiC = self.cal_phiC()
        self.phiS = 1-self.phiC
        self.osm = self.cal_osm()
        self.molar = self.phiC / (self.nu * self.Vs)
        self.molal = self.phiC / (18 * self.phiS * self.nu)*1000
        self.dG, self.dH, self.dS = self.cal_FH_free_energy(), self.cal_FH_enthalpy(), self.cal_FH_entropy()
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        
    def __str__(self):
        return f"Cosol (\u03BD={self.nu}, \u03C7={self.chi}, \u03C7ₛ={self.chiTS})"

    
class Protein(var):
    '''
    Protein class, contains class variable and methods that depend on the protein propeties.

    Args:
        SASA: Change in solvent accesible surface area due to protein folding
    '''
    
    def __init__(self, SASA):
        self.SASA = SASA        
    def __str__(self):
        return f"Protein (SASA={self.SASA})"

class crowding_model(var):
    '''
    Mean-field model class, contains class variable and methods that used to solve the folding thermodynamics of
    a protein-cosolute pair, i.e., for a set of SASA, nu, chi, and eps.

    
    Args:
        protien: A protein class - used to get protein parameters (SASA)
        cosol: A cosolute class - used to get cosolute parameters (nu, chi, chiTS)
        eps: soft interaction parameter
        epsTs: The entropic contribution of eps
        phiC_min: Minimal concetration (volume fraction)
        phiC_max: Maximal concetration (volume fraction)
        dphiC: Concetraion step (volume fraction)  
    '''
    
    def __init__(self, protein, cosol, eps, epsTS, phiC_min=0.0001, phiC_max=0.15, dphiC=0.0001):
        self.nu, self.chi, self.chiTS = cosol.nu, cosol.chi, cosol.chiTS
        self.eps, self.epsTS, self.SASA = eps, epsTS, protein.SASA
        self.chiH, self.epsH = self.chi + self.chiTS, self.eps + self.epsTS
        self.a = self.nu**(1/3)
        self.phiC_min, self.phiC_max, self.dphiC = phiC_min, phiC_max, dphiC
        self.phiC = self.cal_phiC()
        self.phiS = 1-self.phiC
        self.osm = self.cal_osm()
        self.molar = self.phiC / (self.nu * self.Vs)
        self.molal = self.phiC / (18 * self.phiS * self.nu)*1000
        self.muC, self.muS = self.cal_muC(), self.cal_muS()

        self.phiCsurf, self.phiSsurf = np.zeros(self.phiC.shape), np.zeros(self.phiC.shape) #np.nan
        self.flag=False # Enable calculation of thermodynamic potential only after solving the equilibrium condition
        
        self.gamma = 0
        self.ddA, self.ddA_nu, self.ddA_chi, self.ddA_eps = 0, 0, 0, 0
        self.ddE, self.ddE_chi, self.ddE_eps = 0, 0 ,0
        self.TddS, self.TddS_nu, self.TddS_chi, self.TddS_eps = 0, 0, 0 ,0
        
        # _Ms = per protein domain volume "Ms"
        self.gamma_Ms = 0 
        self.ddA_Ms, self.ddA_nu_Ms, self.ddA_chi_Ms, self.ddA_eps_Ms = 0, 0, 0, 0
        self.ddE_Ms, self.ddE_chi_Ms, self.ddE_eps_Ms = 0, 0 ,0
        self.TddS_Ms, self.TddS_nu_Ms, self.TddS_chi_Ms, self.TddS_eps_Ms = 0, 0, 0 ,0
        self.dddA_Ms, self.dddE_Ms, self.TdddS_Ms = 0, 0, 0

        # _kj = in units of kilo joul 
        self.ddA_kj, self.ddA_nu_kj, self.ddA_chi_kj, self.ddA_eps_kj = 0, 0, 0, 0
        self.ddE_kj, self.ddE_chi_kj, self.ddE_eps_kj = 0, 0 ,0
        self.TddS_kj, self.TddS_nu_kj, self.TddS_chi_kj, self.TddS_eps_kj = 0, 0, 0 ,0
        
    def __str__(self):
        return f"Mean-Field Model:\nSoft_Interactions (\u03B5={self.eps}, \u03B5ₛ={self.epsTS}) \nProtein (SASA={self.SASA}) \nCosolute (\u03BD={self.nu}, \u03C7={self.chi}, \u03C7ₛ={self.chiTS})"
 
    def fit_eps(self, exp_conc, exp_ddG, concentration_type='phi'):
        ''' 
        Fit the experimental folding free energy to resolve the soft interaction parameter, eps

        Arg:
            exp_conc: Molar or volume fraction of cosolute in experiment
            exp_ddG: Folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        exp_conc = np.array(exp_conc)
        exp_ddG = np.array(exp_ddG)
        if concentration_type == 'phi':
            pass
        elif concentration_type == 'molar':
            exp_conc = exp_conc * self.nu * self.Vs
        elif concentration_type=='molal':
            exp_conc = exp_conc * (18 * self.phiS * self.nu) * 1000
        else:
            raise Exception("Concetration type can be either molar or volume fraction")
        model_phiC, model_phiS = self.phiC, self.phiS # save model arrays
        # use experiment arrays for the fit
        self.phiC = exp_conc
        self.phiS = 1-self.phiC
        self.phiCsurf = np.zeros(self.phiC.shape)
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.flag=True
        minimize(self.rmsd_fit_eps, self.eps, (exp_ddG), options={'disp':True})
        # return to model arrays
        self.phiC, self.phiS= model_phiC, model_phiS
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.phiCsurf = np.zeros(self.phiC.shape)
        # solve for final eps
        self.solve_crowding()
        self.eps=self.eps[0]
        self.to_pandas()

    def fit_epsTS(self, exp_conc, exp_ddH, exp_TddS, concentration_type='phi'):
        ''' 
        Fit the experimental folding free energy to resolve the soft interaction parameter, eps

        Arg:
            exp_conc: Molar or volume fraction of cosolute in experiment
            exp_ddG: Folding free energy in experiment in kJ/mol
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        exp_conc = np.array(exp_conc)
        exp_ddH = np.array(exp_ddH)
        exp_TddS = np.array(exp_TddS)
        if concentration_type == 'phi':
            pass
        elif concentration_type == 'molar':
            exp_conc = exp_conc * self.nu * self.Vs
        elif concentration_type=='molal':
            exp_conc = exp_conc * (18 * self.phiS * self.nu) * 1000
        else:
            raise Exception("Concetration type can be either molar or volume fraction")
        model_phiC, model_phiS = self.phiC, self.phiS # save model arrays
        # use experiment arrays for the fit
        self.phiC = exp_conc
        self.phiS = 1-self.phiC
        self.phiCsurf = np.zeros(self.phiC.shape)
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.flag=True
        minimize(self.rmsd_fit_epsTS, np.array(self.epsTS), (exp_ddH, exp_TddS), options={'disp':True})
        # return to model arrays
        self.phiC, self.phiS= model_phiC, model_phiS
        self.muC, self.muS = self.cal_muC(), self.cal_muS()
        self.phiCsurf = np.zeros(self.phiC.shape)
        # solve for final eps
        self.solve_crowding()
        self.epsTS=self.epsTS[0]
        self.epsH=self.epsH[0]
        self.to_pandas()
        
    def rmsd_fit_eps(self,eps, ddG):
        self.eps = eps
        self.solve_crowding()
        return (((ddG-self.ddA_kj)**2/len(ddG))**0.5).sum()
    
    def rmsd_fit_epsTS(self,epsTS, ddH, TddS):
        self.epsTS=epsTS
        self.epsH = self.eps + self.epsTS
        self.solve_crowding()
        return (((ddH-self.ddE_kj)**2/len(ddH))**0.5).sum() + (((TddS-self.TddS_kj)**2/len(TddS))**0.5).sum() 
    
    def condition(self,corr_phiCsurf,i):
        ''' 
        Equilibrium condition for fsolve at a given concetration
        
        Args:
            corr_phiCsurf: Corrent value of volume fraction in protein domain
            i: Concetration index
        '''
        return self.cal_muCsurf(corr_phiCsurf) - (self.nu)*self.cal_muSsurf(corr_phiCsurf) - self.muC[i] + (self.nu)*self.muS[i]

    def solve_cond(self, i):
        ''' 
        Solves the equilbrium condition for a given bulk concetration 
        
        Args:
            i: Concetration index
        '''       
        return fsolve(self.condition, (self.phiC[i]), args=(i))[0]

    def solve_crowding(self):
        ''' Solves the equilbrium condition for the entire concetration range'''     
        for i in range(len(self.phiC)):
            self.phiCsurf[i] = self.solve_cond(i)
        self.flag=True
        self.phiSsurf = 1-self.phiCsurf
        self.cal_global_scaling()
        self.cal_glabal_phiCmix()
        self.cal_gamma()
        self.cal_Free_Energy_nu()
        self.cal_Free_Energy_chi()
        self.cal_Free_Energy_eps()
        self.cal_Free_Energy()
        self.cal_Energy_chi()
        self.cal_Energy_eps()
        self.cal_Energy()
        self.cal_Entropy_nu()
        self.cal_Entropy_chi()
        self.cal_Entropy_eps()
        self.cal_Entropy()
        
    def cal_scaling(self, corr_phiCsurf):
        ''' 
        Calculate the scaling factor for a given volume fraction in the protein domain

        Arg:
            corr_phiCsurf: The current cosolute concentration in the protein domain
        '''
        return 1-( 0.5*(1-1/self.a)*(1-corr_phiCsurf) )

    def cal_global_scaling(self):
        ''' 
        Calculate the scaling factor for the entire volume fraction array in the protein domain
        '''
        assert self.flag, 'Run solve_crowding first'
        self.scaling = 1-( 0.5*(1-1/self.a)*(1-self.phiCsurf)) 
                  
    def cal_phiCmix(self, corr_phiCsurf):
        ''' 
        Calculate the scaled volume fraction in the protein domain for a given volume fraction

        Arg:
            corr_phiCsurf: The current cosolute concentration in the protein domain
        '''
        return corr_phiCsurf / self.cal_scaling(corr_phiCsurf)
    
    def cal_glabal_phiCmix(self):
        ''' 
        Calculate the scaled volume fraction in the protein domain for the entire volume fraction array
        '''
        assert self.flag, 'Run solve_crowding first'
        self.phiCmix =  self.phiCsurf / self.scaling
        
    def cal_muCsurf(self, corr_phiCsurf):
        ''' 
        Calculate the protein domain chemical potential of the cosolute for a given volume fraction

        Arg:
            corr_phiCsurf: The current cosolute concentration in the protein domain
        '''
        self.corr_scaling = self.cal_scaling(corr_phiCsurf)
        self.corr_phiCmix = self.cal_phiCmix(corr_phiCsurf)
        term1 = ((self.nu*self.corr_scaling+(0.5-(1/(2*self.a)))*self.nu*(1-corr_phiCsurf)-self.nu))*np.log(1-self.corr_phiCmix)
        term2 = (((self.corr_scaling)-corr_phiCsurf))*((self.nu*(1-corr_phiCsurf))/(1-self.corr_phiCmix))*((((1/2)-(1/(2*self.a)))-1)/((self.corr_scaling)**2)) 
        term3 = np.log(self.corr_phiCmix)
        term4 = (self.corr_phiCmix/corr_phiCsurf)*((1-corr_phiCsurf)*(self.corr_scaling)-corr_phiCsurf*(1-corr_phiCsurf)*((1/2)-(1/(2*self.a)))) 
        term5 = (self.nu)*self.chi*((1-corr_phiCsurf)**2) + (self.eps*self.nu)/(self.a)
        return np.sum([term1,term2,term3,term4,term5])

    def cal_muSsurf(self,corr_phiCsurf):
        ''' 
        Calculate the protein domain chemical potential of the solvent for a given volume fraction

        Arg:
            corr_phiCsurf: The current cosolute concentration in the protein domain
        '''
        self.corr_scaling = self.cal_scaling(corr_phiCsurf)
        self.corr_phiCmix = self.cal_phiCmix(corr_phiCsurf)
        term1 = ((self.corr_scaling-((1/2)-(1/(2*self.a)))*corr_phiCsurf))*np.log(1-self.corr_phiCmix)
        term2 = (self.corr_scaling-corr_phiCsurf)*(corr_phiCsurf/(1-self.corr_phiCmix))*((1-((1/2)-(1/(2*self.a))))/((self.corr_scaling)**2))
        term3 = (1/self.nu)*corr_phiCsurf*((-1+((1/2)-(1/(2*self.a))))/(self.corr_scaling))
        term4 = self.chi*((corr_phiCsurf)**2)
        return np.sum([term1,term2,term3,term4])

    def cal_Free_Energy_nu(self):
        ''' Calculate the contribution of excluded volume to the folding free energy '''
        assert self.flag, 'Run solve_crowding first'
        term1 = ((self.phiC-self.phiCsurf)) + ((1-self.phiCsurf))*np.log(1-self.phiC) 
        term2 = -self.scaling*((1-self.phiCmix))*np.log(1-self.phiCmix)
        term3 = ((self.phiCsurf-self.phiC)/self.nu) + (self.phiCsurf/self.nu)*np.log(self.phiC)
        term4 = -self.scaling*(self.phiCmix/self.nu)*np.log(self.phiCmix)
        self.ddA_nu_Ms = term1+term2+term3+term4
        self.ddA_nu = self.for_all_SASA(self.ddA_nu_Ms)
        self.ddA_nu_kj = self.ddA_nu*self.R*self.T/1000
        
    def cal_Free_Energy_chi(self):
        ''' Calculate the contribution of non-ideal interactions to the folding free energy '''
        assert self.flag, 'Run solve_crowding first'
        term1 = self.chi*(self.phiCsurf-self.phiC)*(1-self.phiC)
        term2 = self.chi*self.phiC*(1-self.phiCsurf) 
        term3 = -self.chi*(1-self.phiCsurf)*self.phiCsurf
        self.ddA_chi_Ms = term1+term2+term3
        self.ddA_chi = self.for_all_SASA(self.ddA_chi_Ms)
        self.ddA_chi_kj = self.ddA_chi*self.R*self.T/1000 

    def cal_Free_Energy_eps(self):
        ''' Calculate the contribution of soft interactions to the folding free energy '''
        assert self.flag, 'Run solve_crowding first'
        self.ddA_eps_Ms =  -(self.eps*self.phiCsurf)/(self.a)
        self.ddA_eps = self.for_all_SASA(self.ddA_eps_Ms)
        self.ddA_eps_kj = self.ddA_eps*self.R*self.T/1000

    def cal_Free_Energy(self):
        ''' Calculate the folding free energy '''
        assert self.flag, 'Run solve_crowding first'
        self.ddA_Ms = self.ddA_nu_Ms + self.ddA_chi_Ms + self.ddA_eps_Ms
        self.ddA = self.ddA_nu + self.ddA_chi + self.ddA_eps
        self.ddA_kj = self.ddA_nu_kj + self.ddA_chi_kj + self.ddA_eps_kj

    def cal_Energy_chi(self):
        ''' Calculate the contribution of non-ideal mixing to the folding energy '''
        assert self.flag, 'Run solve_crowding first'
        term1 = (self.chiH)*(self.phiCsurf-self.phiC)*(1-self.phiC) 
        term2 = (self.chiH)*self.phiC*(1-self.phiCsurf) 
        term3 = -(self.chiH)*(1-self.phiCsurf)*self.phiCsurf
        self.ddE_chi_Ms = term1+term2+term3
        self.ddE_chi = self.for_all_SASA(self.ddE_chi_Ms)
        self.ddE_chi_kj = self.ddE_chi*self.R*self.T/1000
        
    def cal_Energy_eps(self):
        ''' Calculate the contribution of soft interactions to the folding energy '''
        assert self.flag, 'Run solve_crowding first'
        self.ddE_eps_Ms = -((self.epsH)*self.phiCsurf)/(self.a)
        self.ddE_eps = self.for_all_SASA(self.ddE_eps_Ms)
        self.ddE_eps_kj = self.ddE_eps*self.R*self.T/1000

    def cal_Energy(self):
        ''' Calculate the folding energy '''
        assert self.flag, 'Run solve_crowding first'
        self.ddE_Ms = self.ddE_chi_Ms + self.ddE_eps_Ms
        self.ddE = self.ddE_chi + self.ddE_eps
        self.ddE_kj = self.ddE_chi_kj + self.ddE_eps_kj

    def cal_Entropy_nu(self):
        ''' Calculate the contribution of excluded volume to the folding entropy '''
        assert self.flag, 'Run solve_crowding first'
        self.TddS_nu_Ms = -self.ddA_nu_Ms
        self.TddS_nu = -self.ddA_nu
        self.TddS_nu_kj = -self.ddA_nu_kj
        
    def cal_Entropy_chi(self):
        ''' Calculate the contribution of non-ideal mixing to the folding entropy '''
        assert self.flag, 'Run solve_crowding first'
        self.TddS_chi_Ms = self.ddE_chi_Ms-self.ddA_chi_Ms
        self.TddS_chi = self.ddE_chi-self.ddA_chi
        self.TddS_chi_kj = self.ddE_chi_kj-self.ddA_chi_kj
        
    def cal_Entropy_eps(self):
        ''' Calculate the contribution of soft-interactions to the folding entropy '''
        assert self.flag, 'Run solve_crowding first'
        self.TddS_eps_Ms = self.ddE_eps_Ms-self.ddA_eps_Ms
        self.TddS_eps = self.ddE_eps-self.ddA_eps
        self.TddS_eps_kj = self.ddE_eps_kj-self.ddA_eps_kj
        
    def cal_Entropy(self):
        ''' Calculate the folding entropy '''
        assert self.flag, 'Run solve_crowding first'
        self.TddS_Ms = self.TddS_nu_Ms + self.TddS_chi_Ms + self.TddS_eps_Ms
        self.TddS = self.TddS_nu + self.TddS_chi + self.TddS_eps
        self.TddS_kj = self.TddS_nu_kj + self.TddS_chi_kj + self.TddS_eps_kj
    
    def cal_gamma(self):
        ''' Calculate the preferential hydration coefficient '''
        assert self.flag, 'Run solve_crowding first'
        self.gamma_Ms = -(self.phiSsurf*(1 - (self.phiCsurf/self.phiC)*(self.phiS/self.phiSsurf) ))
        self.gamma = self.for_all_SASA(self.gamma_Ms)
        
    def for_all_SASA(self, data):
        ''' 
        Convert potential from value per protein domain volume to value for the entire protein

        Arg:
            data: A thermodynamic potential: Free energy, energy, or entropy.
        '''
        return data*(self.SASA/30**(2/3))*self.a

    def plot_results(self, concentration_type='phi', exp_conc=np.nan, exp_ddG=np.nan, 
                    exp_concT=np.nan, exp_ddH=np.nan, exp_TddS=np.nan):
        ''' 
        Plot model results 

        Arg:
            concentration_type: type of concetration. str - 'phi', 'molar', or 'molal'
        '''
        assert self.flag, 'Run solve_crowding first'
        if concentration_type == 'phi':
            conc = self.phiC
            str_conc = r'$\phi_C$'
        elif concentration_type=='molar':
            conc = self.molar
            str_conc = 'molar'
        elif concentration_type=='molal':
            conc = self.molal
            str_conc = 'molal'

            
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8, 8), layout="constrained")
        axes[0,0].plot(conc, self.gamma)
        axes[0,0].set_xlabel(str_conc)
        axes[0,0].set_ylabel(r'$\Delta\Gamma_S$')

        axes[0,1].plot(conc, self.osm)
        axes[0,1].set_xlabel(str_conc)
        axes[0,1].set_ylabel(r'$\Pi (Osmolal)$')

        axes[0,2].plot(conc, self.phiCsurf)
        axes[0,2].set_xlabel(str_conc)
        axes[0,2].set_ylabel(r'$\phi_C^{surf}$')

        axes[1,0].plot(conc, self.ddA_kj)
        axes[1,0].plot(conc, self.ddA_nu_kj)
        axes[1,0].plot(conc, self.ddA_chi_kj)
        axes[1,0].plot(conc, self.ddA_eps_kj)
        axes[1,0].plot(exp_conc, exp_ddG,'o', label='_nolegend_')
        axes[1,0].set_xlabel(str_conc)
        axes[1,0].set_ylabel(r'$\Delta\Delta G_i^{0}$')
        axes[1,0].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
        
        axes[1,1].plot(conc, self.ddE_kj)
        axes[1,1].plot(conc, self.ddE_chi_kj)
        axes[1,1].plot(conc, self.ddE_eps_kj)
        axes[1,1].plot(exp_concT, exp_ddH,'o', label='_nolegend_')   
        axes[1,1].set_xlabel(str_conc)
        axes[1,1].set_ylabel(r'$\Delta\Delta H_i^{0}$')
        axes[1,1].legend(['tot',r'$\chi$',r'$\varepsilon$'])

        axes[1,2].plot(conc, self.TddS_kj)
        axes[1,2].plot(conc, self.TddS_nu_kj)
        axes[1,2].plot(conc, self.TddS_chi_kj)
        axes[1,2].plot(conc, self.TddS_eps_kj)
        axes[1,2].plot(exp_concT, exp_TddS,'o', label='_nolegend_')   
        axes[1,2].set_xlabel(str_conc)
        axes[1,2].set_ylabel(r'$T\Delta\Delta S_i^{0}$')
        axes[1,2].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])

        axes[2,0].plot(self.osm, self.ddA_kj)
        axes[2,0].plot(self.osm, self.ddA_nu_kj)
        axes[2,0].plot(self.osm, self.ddA_chi_kj)
        axes[2,0].plot(self.osm, self.ddA_eps_kj)
        axes[2,0].set_xlabel(r'$\Pi (Osmolal)$')
        axes[2,0].set_ylabel(r'$\Delta\Delta G_i^{0}$')
        axes[2,0].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])

        axes[2,1].plot([-max(abs(self.ddE_kj)),max(abs(self.ddE_kj))], [-max(abs(self.ddE_kj)),max(abs(self.ddE_kj))], color="darkgrey",label='_nolegend_') 
        axes[2,1].plot([-max(abs(self.ddE_kj)),max(abs(self.ddE_kj))], [max(abs(self.ddE_kj)),-max(abs(self.ddE_kj))], color="darkgrey",label='_nolegend_')
        axes[2,1].plot(self.ddE_kj, self.TddS_kj)
        axes[2,1].plot(np.zeros(self.TddS_nu_kj.shape), self.TddS_nu_kj)
        axes[2,1].plot(self.ddE_chi_kj, self.TddS_chi_kj)
        axes[2,1].plot(self.ddE_eps_kj, self.TddS_eps_kj)
        axes[2,1].plot(exp_ddH, exp_TddS,'o', label='_nolegend_')   
        axes[2,1].set_xlabel(r'$\Delta\Delta H_i^{0}$')
        axes[2,1].set_ylabel(r'$T\Delta\Delta S_i^{0}$')
        axes[2,1].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])

        if max(abs(self.ddE_chi_kj)) != 0:
            axes[2,2].set_xlim([-max(abs(self.ddE_chi_kj)),max(abs(self.ddE_chi_kj))])
        else:
            axes[2,2].set_xlim([-max(abs(self.TddS_chi_kj)),max(abs(self.TddS_chi_kj))])
        axes[2,2].set_ylim([-max(abs(self.TddS_chi_kj)),max(abs(self.TddS_chi_kj))])
        axes[2,2].plot([-max(abs(self.ddE_kj)),max(abs(self.ddE_kj))], [-max(abs(self.ddE_kj)),max(abs(self.ddE_kj))], color="darkgrey",label='_nolegend_') 
        axes[2,2].plot([-max(abs(self.ddE_kj)),max(abs(self.ddE_kj))], [max(abs(self.ddE_kj)),-max(abs(self.ddE_kj))], color="darkgrey",label='_nolegend_')
        axes[2,2].plot(self.ddE_kj, self.TddS_kj)
        axes[2,2].plot(np.zeros(self.TddS_nu_kj.shape), self.TddS_nu_kj)
        axes[2,2].plot(self.ddE_chi_kj, self.TddS_chi_kj)
        axes[2,2].plot(self.ddE_eps_kj, self.TddS_eps_kj)
        axes[2,2].set_xlabel(r'$\Delta\Delta H_i^{0}$')
        axes[2,2].set_ylabel(r'$T\Delta\Delta S_i^{0}$')
        axes[2,2].legend(['tot',r'$\nu$',r'$\chi$',r'$\varepsilon$'])
        axes[2,2].locator_params(axis='both', nbins=3)
        plt.show()

    def to_pandas(self):
        self.results = pd.DataFrame({'phiC':self.phiC,
                            'phiCsurf':self.phiCsurf, 
                            'phiS':self.phiS, 
                            'phiSsurf':self.phiSsurf, 
                            'molar':self.molar,
                            'osm':self.osm,
                            'gamma_per_vol':self.gamma_Ms, 
                            'gamma':self.gamma,
                            'ddA_nu':self.ddA_nu, 
                            'ddA_chi':self.ddA_chi, 
                            'ddA_eps':self.ddA_eps, 
                            'ddA':self.ddA,
                            'ddE_chi':self.ddE_chi, 
                            'ddE_eps':self.ddE_eps, 
                            'ddE':self.ddE,
                            'TddS_nu':self.TddS_nu, 
                            'TddS_chi':self.TddS_chi, 
                            'TddS_eps':self.TddS_eps, 
                            'TddS':self.TddS,
                            'ddA_nu_kJ':self.ddA_nu_kj, 
                            'ddA_chi_kJ':self.ddA_chi_kj, 
                            'ddA_eps_kJ':self.ddA_eps_kj, 
                            'ddA_kJ':self.ddA_kj,
                            'ddE_chi_kJ':self.ddE_chi_kj, 
                            'ddE_eps_kJ':self.ddE_eps_kj, 
                            'ddE_kJ':self.ddE_kj,
                            'TddS_nu_kJ':self.TddS_nu_kj, 
                            'TddS_chi_kJ':self.TddS_chi_kj, 
                            'TddS_eps_kJ':self.TddS_eps_kj, 
                            'TddS_kJ':self.TddS_kj})
    
        