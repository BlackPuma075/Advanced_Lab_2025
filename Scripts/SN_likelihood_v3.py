import numpy as np
from astropy.cosmology import FlatLambdaCDM
import emcee
from astropy import units as u
from mpi4py import MPI
from schwimmbad import MPIPool
import sys
import time

start_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def Covariance(C_eta, alpha, beta):
    A = np.zeros((740,3*740)) #Cambiar la implementación de A por producto tensorial (I*\vec{A})
    for i in range(740):
        A[i, 3*i] = 1
        A[i, 3*i +1] = alpha
        A[i, 3*i +2] = beta
    covariance = np.matmul(A.T,np.matmul(C_eta,A))
    
    return covariance, A

def eta_matrix(X_1,C,m_b):
    eta = np.zeros((3*740))
    for i in range(740):
        eta[3*i] = X_1[i]
        eta[3*i +1] = C[i]
        eta[3*i +2] = m_b[i]
    return eta


def lumdist(z, Om,H0):
    new_z = np.zeros(3*len(z))
    for i in range(len(z)):
        new_z[3*i] = z[i] #Modificar, sólo es un vector de (740)
        new_z[3*i + 1] = z[i]
        new_z[3*i + 2] = z[i]

    cosmo = FlatLambdaCDM(H0=H0, Om0=Om, Tcmb0=2.725)

    return  cosmo.luminosity_distance(new_z)

#Añadir función para añadir o no delta M_B (Stellar mass '3rdvar' en dataset)

def log_likelihood(theta, x, cov):
    M_B, dM_B, alpha, beta, Om, H0 = theta #varied parameters
    #Añadir M_B = M_B  M_{stellar}<10^{10}M_{sun} and M_B = M_B + delta M_B otherwise
    covariance, A = Covariance(cov, alpha, beta)
    cov = np.linalg.pinv(covariance) #covariance matrix
    z, X_1, C, m_b = x #observed data
    eta = eta_matrix(X_1, C, m_b)
    mu = (1+alpha-beta)*eta-M_B #REVISAR DIMENSIONES!!!!!
    #model = m_b-(M_B-alpha*X_1+beta*C) #model
    media = lumdist(z, Om, H0)
    media = media.to_value(u.Mpc)
    vec = (mu - media) #data vector
    Xi = np.matmul(np.matmul(vec.T, cov),vec) #computation of the xi_sqrd
    
    return -0.5*Xi

def log_prior(theta):
    M_B, dM_B, alpha, beta, Om,H0 = theta
    if -30 < M_B < 0 and -1.0 < dM_B < 1.0 and -1.0 < alpha < 1.0 and 0.0 < beta < 5.0 and 0.0 < Om < 0.5 and 60 < H0 < 80: #priors based on https://arxiv.org/pdf/1401.4064
        return 0.0
    return -np.inf

def log_probability(theta, x, cov):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, cov)


C_eta = np.load('C_eta.npy')


ndim = 6
nwalkers = 12
p0 = np.zeros((12,6))


for i in range(6):
    for j in range(12):
        mean =[-20,0,0,2,0.3,70.] #Revisar guess
        std = [0.1,0.01, 0.01, 0.01, 0.01, 0.1] #Revisar dispersión   
        p0[j,i] = np.random.normal(mean[i], std[i])


import pandas as pd
data = pd.read_csv('jla_likelihood_v6/jla_likelihood_v6/data/jla_lcparams.txt', sep='\s+')



x = [data['zcmb'],data['x1'],data['color'],data['mb']]

# Usar MPIPool para paralelización
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[x, C_eta], pool=pool)

    #Corrida inicial (10 steps)
    state = sampler.run_mcmc(p0, 10, progress=True)[0]

    sampler.reset()

    all_chains = []
    
    #Loop para guardar cada 100 iteraciones
    for i in range(10):
        sampler.run_mcmc(state, 1000, progress=True)
    
        if rank == 0: 
            chain = sampler.get_chain()
            all_chains.append(chain.copy())
    
            #Guardar un checkpoint de la cadena acumulada
            np.save(f"chain_checkpoint_{(i+1)*1000}.npy", np.concatenate(all_chains, axis=0))
            print(f"Checkpoint guardado en chain_checkpoint_{(i+1)*1000}.npy")
    
    #Guardar la cadena completa
    if rank == 0:
        np.save("chain_complete.npy", np.concatenate(all_chains, axis=0))
        print("Cadena completa guardada en 'chain_complete.npy'.")
    
    print("MCMC completado.")

        
end_time = time.time()  #Tiempo de finalización
elapsed_time = end_time - start_time  #Tiempo total

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60

print(f"Tiempo total de ejecución: {hours}h {minutes}m {seconds:.2f}s")        

