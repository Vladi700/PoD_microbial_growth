import numpy as np
from scipy.stats import gamma,beta
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt
import pandas as pd

#gamma:
def f_alpha_i(params,alpha):
    tau, k, alpha_S, alpha2, beta2, alpha1, scale1 = params
    return gamma.pdf(alpha,a=alpha1,scale=scale1)

#beta:
def f_f_i(params,f_i):
    tau, k, alpha_S, alpha2, beta2, alpha1, scale1 = params
    return beta.pdf(f_i,a=alpha2,b=beta2)

''' Logarithm of likelihood of all measurement'''

def log_likelihood(params, initial_mass, lineages, t, alpha, f_i):
    tau, k, alpha_S, alpha2, beta2, alpha1, scale1 = params
    masses_before_division = np.array([])

    """for j, i in enumerate(np.unique(lineages)):
        filtered_times = t[lineages == i]
        filtered_alphas = alpha[lineages == i]
        filtered_fs = f_i[lineages == i]
        masses_before_division_per_lineage = evolve_mass(params, filtered_times, filtered_alphas, filtered_fs, initial_mass[j])
        masses_before_division = np.concatenate((masses_before_division, masses_before_division_per_lineage))"""
    masses_before_division = initial_mass
    first = p(t, tau, k, alpha_S, initial_mass, alpha) #change
    second = f_alpha_i(params, alpha)
    third = f_f_i(params, f_i)

    first = np.where(first == 0, 1e-300, first)
    second = np.where(second == 0, 1e-300, second)
    third = np.where(third == 0, 1e-300, third)

    res = np.sum(np.log(first) + np.log(second) + np.log(third))
    #res = np.sum(np.log(first))

    return res

'''Hazard function (probability of having a division with certain parameters)'''

#in generation data m_D changes every time wherease in fit data it is a constant=initial_mass

def protein(t,m_D, alpha):
    return m_D *( np.exp(alpha*t) - 1)


"""def h(t, tau, k, alpha_S, m_D, alpha, mode="generation"):
    p = protein(t, m_D, alpha)
    dpdt = m_D * alpha * np.exp(alpha * t)

    # Core hazard function in terms of protein
    hazard_core = (alpha_S * k / tau) * ((p / tau)**(k - 1)) / (1 + (p / tau)**k)
    h_t = hazard_core * dpdt

    if mode == "generation":
        return h_t
    elif mode == "inference":
        return h_t"""

    
"""def s(t, tau, k, alpha_S, m_D, alpha, mode="generation"):
    p = protein(t, m_D, alpha)
    S_p = 1 / (1 + (p / tau)**k)**alpha_S
    if mode == "generation":
        # If you want to introduce a t0 cutoff as before, do it here
        return S_p
    elif mode == "inference":
        return S_p"""

"""def h(t, tau, k, alpha_S, m_D, alpha, mode="generation"):
    p = protein(t, m_D, alpha)
    dpdt = m_D * alpha * np.exp(alpha * t)
    hazard_core = (alpha_S * k / tau) * ((p / tau) ** (k - 1)) / (1 + (p / tau) ** k)
    return hazard_core * dpdt


def s(t, tau, k, alpha_S, m_D, alpha, mode="generation"):
    p = protein(t, m_D, alpha)
    return 1 / (1 + (p / tau) ** k) ** alpha_S"""

def s(t, tau, k, alpha_gl, m_D, alpha):
    p = protein(t, m_D, alpha)
    return 1 / (1 + (p / tau)**k)**alpha_gl

def h(t, tau, k, alpha_gl, m_D, alpha):
    p = protein(t, m_D, alpha)
    term = p / tau
    dpdt = alpha * m_D * np.exp(alpha * t)  # derivative of p(t)
    
    hazard_core = (alpha_gl * k / tau) * (term**(k - 1)) / (1 + term**k)
    return hazard_core * dpdt

   

############################################################################
#used only for generating synthetic data:
'''Survival function used for the sampling (and only there)'''

def s_root(t, tau, k, alpha_S, m_D, alpha):
    return s(t, tau, k, alpha_S, m_D, alpha)


def t0_root(tau, m_D, alpha):
    res = (tau / m_D) + 1
    return np.log(res) / alpha

''' Logarithm of prior function '''
#def log_prior(params):
#    tau, k, alpha_S, alpha2, beta2, alpha1, scale1 = params
#    if alpha2 > 0 and beta2 > 0 and alpha1 > 0 and scale1 > 0 and alpha_S > 0 and tau > 0 and tau < 20 and k > tau and k < 20:
#        return 0.0
#    else:
#        return -np.inf

def log_prior(params):
    tau, k, alpha_S, alpha2, beta2, alpha1, scale1 = params
    
    valid = (
        0 < tau <= 10 and
        1 < k <= 10 and
        0 < alpha_S <= 10 and
        alpha2 > 0 and
        beta2 > 0 and
        alpha1 > 0 and
        scale1 > 0
    )
    
    return 0.0 if valid else -np.inf



def log_probability(params, initial_mass, lineages, t, alpha, f_i):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    res = lp + log_likelihood(params, initial_mass, lineages, t, alpha, f_i)
    return res

############################################################################

'''Distribution of times of divisions'''
#is the one that we have called first in log_loikelihhod function
#def p(t, tau, k, alpha_S, m_D, alpha):
#    return h(t, tau, k, alpha_S, m_D, alpha) * s(t, tau, k, alpha_S, m_D, alpha)
    
def p(t, tau, k, alpha_gl, m_D, alpha):

    return h(t, tau, k, alpha_gl, m_D, alpha) * s(t, tau, k, alpha_gl, m_D, alpha)





'''Mass at division'''
def m(t, m_D, alpha):
    return m_D * np.exp(alpha * t)


def t0(params, m_D, alpha):
    tau, k, alpha_S, alpha2, beta2, alpha1, scale1 = params
    res = (tau / m_D) + 1
    return np.log(res) / alpha


def evolve_mass(params, filtered_times, filtered_alphas, filtered_fs, initial_mass):
    masses = np.empty(filtered_times.shape[0])
    mass = initial_mass
    for i in range(masses.shape[0]):
        masses[i] = mass
        mass = m(filtered_times[i], mass, filtered_alphas[i]) * filtered_fs[i]
    return masses



''' Chain plotter '''
#To plot emcee outputs(evolution of parameter values found by random walkers)

def chainplot(chains, interval, parameter, n_plotted=6, random_pick=False, discard=200, thin=25, legend=False):
    
    n_walkers = chains.shape[1]
    
    if random_pick == True:
        plotted_walkers = np.random.choice(np.arange(n_walkers), n_plotted, replace=False)
    else:
        plotted_walkers = np.arange(0, n_plotted)

    return plotted_walkers



#### example, steal this box
#interval = np.arange(3500, chain.shape[0])
#model_2.chainplot(sampler=sampler,       # ensemble sampler from emcee, after it is run
#                  interval=interval,     # which interval of steps is plotted  
#                  parameter = 7,         # which parameter evolution is plotted (counting from 0)
#                  n_plotted = 5,         # default 6 --- number of plotted walkers
#                  random_pick = True,    # default False --- if true pick the walkers randomly, if false the first n_plotted ones 
#                  discard=100,           # default 100 --- parameter for get_chain method, discard the first entries
#                  thin=1,                # default 1 --- pick a sample every n (n=1 -> pick all samples)
#                  legend=True            # default False --- if true, plot legend
#                 )
####


''' Chain sampler'''

def chainsampler (sampler, params, n_samples, n_walkers=1, discard=0, make_hist=False, save_data = False):
    chains = sampler.get_chain(discard = 0)
    chain_length = chains.shape[0]
    
    
    sampled_walker_ids = np.random.choice(np.arange(chains.shape[1]), n_walkers, replace=False)
    chains_picked = chains[:,sampled_walker_ids,:]
    steps_to_pick = np.linspace(0, chain_length, n_samples, dtype=int)
    steps_to_pick = np.delete(steps_to_pick, -1)
    
    if make_hist==True:
        plt.hist(chains_picked[steps_to_pick])
        plt.show
    
    print('Picked walkers id: ', str(sampled_walker_ids))
    
    if save_data==True:
        df = pd.DataFrame(chains_picked[steps_to_pick])
        df.to_csv('chains_ids.csv', index=False)  
        
    return(chains_picked[steps_to_pick,:,:])  

'''Generate synthetic data from an initial mass'''

def evolve(initial_mass=None, masses_list=None, n_cells=1, n_times=10, params=None, t_max=30.0):
  '''Generates synthetic data of n_times divisions for n_cells lineages, all with the same initial mass
    The params are u, v, omega2, alpha1, scale1, alpha2, beta2
    t_max is the max time of division used as a treshold for the root finding algorithm"
  '''
  u, v, omega2,alpha2, beta2,alpha1, scale1 = params
  
  data = {'lineage_ID': [], 'generation_N':[], 'mass_birth':[], 'time_division':[], 'alpha':[], 'f':[]}

  for j in range(n_cells): #For each lineage of cells

    #At the beginning of each lineage the initial mass is the same of the progenitor
    mass = masses_list[j] if initial_mass==None else initial_mass
      
    for i in range(n_times):  #Follow the divisions
      data['generation_N'].append(i)
      data['lineage_ID'].append(j)
      #Generate alpha and f from the respective distributions and store the values in the list
      alpha_D = gamma.rvs(a=alpha1,scale=scale1)  #DO NOT set size=1 because it returns an array instead of a number
      data['alpha'].append(alpha_D)

      f_D=beta.rvs(a=alpha2,b=beta2)
      data['f'].append(f_D)

      #generate random division time
      val = np.random.uniform()
      f_to_root = lambda t, val: 1 - s_root(t,u,v,omega2,mass,alpha_D) - val 
      t_0 = t0_root(u, mass, alpha_D)

      try:
            t_D = root_scalar(f_to_root, bracket=(t_0, t_max),  args=(val,), method='brentq').root 
      except ValueError: 
            t_D = t_max
      data['time_division'].append(t_D)

      #Get mass before division (obtained from previous cycle mass )
      data['mass_birth'].append(mass)
      m_before_D = m(t_D, mass, alpha_D) #where alpha is sampled from Gamma distribution
      #Get mass after division (sample f randomly from Beta distr)
      m_after_D = f_D*m_before_D    #<-where f is sampled from Beta distribution
      mass = m_after_D
    
    # Save masses and times in a global list and use the last masses as starting point for the next step

  return pd.DataFrame(data)

'''Survival function used for the sampling (and only there)'''

# Modified s_root to use the new 's' function and its parameters
def s_root(t, tau, k, alpha_S, m_D, alpha):
    return s(t, tau, k, alpha_S, m_D, alpha)

# Modified t0_root to use 'tau' instead of 'u'
def t0_root(tau, m_D, alpha):
    res = (tau / m_D) + 1
    return np.log(res) / alpha