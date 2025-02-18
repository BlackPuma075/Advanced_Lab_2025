#!/usr/bin/env python
# coding: utf-8

# Model:
# 
# $\mu = m_B^{*}-\left(M_B-\alpha\times X_1 +\beta\times C\right)$
# 
# $\mu$: distance modulus
# 
# $m_B^{*}$: peak magnitude in rest frame
# 
# $M_B$: absolute magnitude
# 
# $\alpha, \beta$: bias parameters
# 
# $C$: color

# In[24]:


import numpy as np
from astropy.cosmology import FlatLambdaCDM
import emcee
from astropy import units as u


# In[46]:


def Covariance(C_eta, alpha, beta):
    I = np.identity(740)
    a_vec = (1,alpha,beta)
    A = np.tensordot(I,a_vec,axes=0).reshape((740,2220))
    covariance = np.matmul(A.T,np.matmul(C_eta,A))
    
    return covariance, A

def A_matrix(alpha,beta):
    I = np.identity(740)
    a_vec = (1,alpha,beta)
    A = np.tensordot(I,a_vec,axes=0).reshape((740,2220))

    return A

def eta_matrix(X_1,C,m_b):
    eta = np.zeros((3*740))
    for i in range(740):
        eta[3*i] = X_1[i]
        eta[3*i +1] = C[i]
        eta[3*i +2] = m_b[i]
    return eta


def lumdist(z, Om,H0):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om, Tcmb0=2.725)

    return  cosmo.luminosity_distance(z)

def lumvec(stellar_mass):
    dM_B = np.ones_like(stellar_mass)
    for i in range(len(stellar_mass)):
        if stellar_mass[i]<10:
            dM_B[i] = 0
        
    return dM_B     

def log_likelihood(theta, cov, lumvec, eta, z):
    M_B, dM_B, alpha, beta, Om, H0 = theta #Varied parameters
    #covariance, A = Covariance(cov, alpha, beta)
    #cov = np.linalg.pinv(covariance) #Covariance matrix
    A = A_matrix(alpha, beta)
    a_vec = np.tile([1, alpha, beta], 2220//3)
    covariance = np.matmul(A.T,np.matmul(cov,A))
    model =a_vec*eta-(M_B*np.ones(2220)+dM_B*np.tile(lumvec,3)) #Model
    mu = lumdist(z, Om, H0)
    mu = mu.to_value(u.Mpc)
    vec = (model - mu) #Data vector
    Xi = np.matmul(np.matmul(vec, covariance),vec.T) #computation of the xi_sqrd
    
    return -0.5*Xi

def log_prior(theta):
    M_B, dM_B, alpha, beta, Om,H0 = theta
    if -30 < M_B < 0 and -1.0 < dM_B < 1.0 and -1.0 < alpha < 1.0 and 0.0 < beta < 5.0 and 0.0 < Om < 0.5 and 65 < H0 < 75: #priors based on https://arxiv.org/pdf/1401.4064 best fit values
        return 0.0
    return -np.inf

def log_probability(theta, cov, lumvec, eta, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, cov, lumvec, eta, z)


# In[47]:


#Data
import pandas as pd
data = pd.read_csv('jla_likelihood_v6/jla_likelihood_v6/data/jla_lcparams.txt', sep='\s+')
C_eta = np.load('C_eta.npy')
luvec = lumvec(data['3rdvar']) #Vector de \Delta{M_B}
eta = eta_matrix(data['x1'],data['color'],data['mb'])
z = np.repeat(data['zcmb'],3)


# In[48]:


ndim = 6
nwalkers = 12
p0 = np.zeros((12,6))

for i in range(6):
    for j in range(12):
        mean =[-20,0,0,2,0.3,70.]
        std = [0.01,0.01, 0.01, 0.01, 0.01, 0.01]  
        p0[j,i] = np.random.normal(mean[i], std[i])


# In[49]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[C_eta, luvec, eta, z])


# In[50]:


get_ipython().run_line_magic('time', 'state = sampler.run_mcmc(p0, 10)')
sampler.reset()


# In[ ]:


get_ipython().run_line_magic('time', 'sampler.run_mcmc(state, 500);')
chain = sampler.get_chain()
np.save("chain_test_full_cov.npy", chain)  # Guarda la cadena en un archivo .npy

