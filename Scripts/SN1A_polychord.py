import glob
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import emcee
from astropy import units as u
import seaborn as sns
from mpi4py import MPI
from schwimmbad import MPIPool
import sys
import time
import getdist
import IPython
from getdist import plots, MCSamples
from numpy import pi, log
import pypolychord
from pypolychord.priors import UniformPrior
try:
    from mpi4py import MPI
except ImportError:
    pass


# Define a four-dimensional spherical gaussian likelihood,
#  width sigma=0.1, centered on the 0 with one derived parameter.
#  The derived parameter is the squared radius

nDims = 5
nDerived = 1
sigma = 0.1

#def likelihood(theta):
#    """ Simple Gaussian Likelihood"""

#    nDims = len(theta)
#    r2 = sum(theta**2)
#    logL = -log(2*pi*sigma*sigma)*nDims/2.0
#    logL += -r2/2/sigma/sigma

#    return logL, [r2]

def loadsqmat(filename):
    dat = np.loadtxt(f'{filename}')
    n = int(dat[0])
    mat = dat[1:].reshape(n, n)
    return mat

#Data
data = pd.read_csv('jla_likelihood_v6/jla_likelihood_v6/data/jla_lcparams.txt', sep=r'\s+') #Datos observacionales de https://arxiv.org/pdf/1401.4064
C_eta = sum([fits.getdata(mat) for mat in glob.glob('C*.fits')]) #Matriz para calcular C_eta
z = data['zcmb'] #Datos de redshift para cada supernova

#Función para calcular c_stat
def c_stat(n): #Número de SN
    #Datos de la matriz de covarianza
    c00 = loadsqmat('jla_v0_covmatrix.dat')
    c11 = loadsqmat('jla_va_covmatrix.dat')
    c22 = loadsqmat('jla_vb_covmatrix.dat')
    c01 = loadsqmat('jla_v0a_covmatrix.dat')
    c02 = loadsqmat('jla_v0b_covmatrix.dat')
    c12 = loadsqmat('jla_vab_covmatrix.dat')
    c = np.zeros((3 * n, 3 * n)) #Generamos un array de (2220,2220)
    #loop para crear la matriz iterando sobre todas las entradas de la matriz
    for i in range(n):
        for j in range(n):
            c[3 * i + 2, 3 * j + 2] = c00[i, j]
            c[3 * i + 1, 3 * j + 1] = c11[i, j]
            c[3 * i, 3 * j] = c22[i, j]
    
            c[3 * i + 2, 3 * j + 1] = c01[i, j]
            c[3 * i + 2, 3 * j] = c02[i, j]
            c[3 * i, 3 * j + 1] = c12[i, j]
    
            c[3 * j + 1, 3 * i + 2] = c01[i, j]
            c[3 * j, 3 * i + 2] = c02[i, j]
            c[3 * j + 1, 3 * i] = c12[i, j]

    return c
    
#ceta = np.load('ceta.npy')
ceta = c_stat(740)

#Función para calcular la matriz Cmu tomada de https://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html
def mu_cov(alpha, beta,Ceta):
    Cmu = np.zeros_like(Ceta[::3, ::3])
    
    coefficients = [1., alpha, -beta]
    for i, coef1 in enumerate(coefficients):
        for j, coef2 in enumerate(coefficients):
            Cmu += (coef1 * coef2) * Ceta[i::3, j::3]
    
    sigma = np.loadtxt('./sigma_mu.txt')
    sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
    Cmu[np.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2
    
    return Cmu

#Función para calcular la matriz A de alphas y betas
def A_matrix(alpha,beta):
    I = np.identity(740) #Creamos una matriz identidad 
    a_vec = (1,alpha,-beta) #Creamos un vector de datos para multiplicar los 1s de I
    A = np.tensordot(I,a_vec,axes=0).reshape((740,2220)) #Convertir las entradas vectoriales de I en 3 dimensiones más

    return A

#Función para calcular el vector de datos eta (m_b1, X_1, C_1, ..., m_bn, X_n, C_n)
def eta_matrix(X_1,C,m_b):
    eta = np.zeros((3*740))
    for i in range(740):
        eta[3*i] = m_b[i]
        eta[3*i +1] = X_1[i]
        eta[3*i +2] = C[i]
    return eta
eta = eta_matrix(data['x1'],data['color'],data['mb']) #Datos de SN

#Función para calcular la distancia luminosa 
def lumdist(z, Om):
    cosmo = FlatLambdaCDM(H0=70, Om0=Om, Tcmb0=2.725) #Fijamos H0 a 70 km/s/Mpc y variamos Omega_m

    return  cosmo.luminosity_distance(z)

#Función que crea un vector de 0s y 1s siguiendo la ecuación 5 de https://arxiv.org/pdf/1401.4064
def lumvec(stellar_mass):
    dM_B = np.ones_like(stellar_mass)
    for i in range(len(stellar_mass)):
        if stellar_mass[i]<10:
            dM_B[i] = 0
        
    return dM_B     
    
luvec = lumvec(data['3rdvar']) #Vector de \Delta{M_B}

#Función de likelihood
#def likelihood(theta, lumvec, eta, z):
def likelihood(theta):    
    M_B, dM_B, alpha, beta, Om = theta #Varied parameters
    A = A_matrix(alpha,beta) #Matriz A
    #covariance = mu_cov(alpha,beta) #Cálculo de la covarianza C_eta
    covariance = mu_cov(alpha, beta, C_eta) #Cálculo de la covarianza completa
    model = (A @ eta) - (M_B * np.ones(740) + dM_B * luvec) #Modelo
    mu = 5 * np.log10(lumdist(z, Om).to_value(u.Mpc)) + 25 #Modelo 'estándar'
    vec = (model - mu) #Data vector
    Xi = vec.T @ np.linalg.inv(covariance) @ vec #Cálculo del Xi^{2}
  
    return -0.5 * Xi, sum(np.array(theta**2)) #Regresa -1/2*(Xi^{2}) y r^2

#Función para priors
def log_prior(theta):
    M_B, dM_B, alpha, beta, Om = theta
    
    #Priors uniformes para todos los parámetros
    if (-21.0 < M_B < -17.0 and -1.0 < dM_B < 1.0 and -1.0 < alpha < 1.0 and 0.0 < beta < 5.0 and 0.2 < Om < 0.4):#priors based on https://arxiv.org/pdf/1401.4064 best fit values
        return 0.0
    else:   
        return -np.inf  #Rechazar si no está dentro de estos límites

#Función para el cálculo del posterior
def log_probability(theta,lumvec, eta, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf #Se rechaza si no es finita
    return lp + log_likelihood(theta, lumvec, eta, z) #Se considera si es finita


# Define a box uniform prior from -1 to 1

def prior(hypercube):
    """ Map [0,1]^D unit hypercube to parameter ranges. """
    M_B = -21.0 + hypercube[0] * ( -17.0 - (-21.0) )   # [-21, -17]
    dM_B = -1.0 + hypercube[1] * ( 1.0 - (-1.0) )      # [-1, 1]
    alpha = -1.0 + hypercube[2] * ( 1.0 - (-1.0) )     # [-1, 1]
    beta = 0.0 + hypercube[3] * ( 5.0 - 0.0 )         # [0, 5]
    Om = 0.2 + hypercube[4] * ( 0.4 - 0.2 )           # [0.2, 0.4]

    
    return np.array([M_B, dM_B, alpha, beta, Om])

# Optional dumper function giving run-time read access to
#  the live points, dead points, weights and evidences

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

# Parameter names
#! This is a list of tuples (label, latex)
#! Derived parameters should be followed by a *

paramnames = [('M_B', r'$M_B$'),
              ('dM_B', r'$\Delta M_B$'),
              ('alpha', r'$\alpha$'),
              ('beta', r'$\beta$'),
              ('Omega_m', r'$\Omega_m$'),
              ('r', 'r*')]


# Run PolyChord
output = pypolychord.run(
    likelihood,
    nDims=5,
    nDerived=1,
    prior=prior,
    dumper=dumper,
    file_root='gaussian_2',
    nlive=20,
    do_clustering=True,
    read_resume=False,
    paramnames=paramnames,
)


from anesthetic import NestedSamples

# Cargar muestras con anesthetic
samples = NestedSamples(root='gaussian_2')

# Graficar
fig, axes = samples.plot_2d(['M_B', 'dM_B', 'alpha', 'beta', 'Omega_m'])
fig.savefig('posterior_sn1a_2.png')


