import glob
import pandas as pd
from astropy.io import fits
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from cobaya.likelihood import Likelihood

class MySupernovaLikelihood(Likelihood):
    
    def initialize(self):
        """Inicializa los datos y matrices necesarias."""
        # Datos observacionales
        self.data = pd.read_csv('jla_likelihood_v6/jla_likelihood_v6/data/jla_lcparams.txt', sep='\s+')

        # Matriz de covarianza C_eta
        self.C_eta = sum([fits.getdata(mat) for mat in glob.glob('C*.fits')])

        # Redshifts de supernovas
        self.z = self.data['zcmb']

        # Vector de ∆M_B (depende de la masa estelar)
        self.lumvec = self.compute_lumvec(self.data['3rdvar'])

        # Vector de datos eta
        self.eta = self.compute_eta_matrix(self.data['x1'], self.data['color'], self.data['mb'])
    
    def compute_lumvec(self, stellar_mass):
        """Crea el vector de ∆M_B según la masa estelar."""
        return np.where(stellar_mass < 10, 0, 1)
    
    def compute_eta_matrix(self, X_1, C, m_b):
        """Crea el vector de datos eta (m_b, X_1, C)."""
        eta = np.zeros(3 * len(m_b))
        for i in range(len(m_b)):
            eta[3 * i] = m_b[i]
            eta[3 * i + 1] = X_1[i]
            eta[3 * i + 2] = C[i]
        return eta

    def compute_mu_cov(self, alpha, beta):
        """Calcula la matriz de covarianza C_mu."""
        Cmu = np.zeros_like(self.C_eta[::3, ::3])
        coefficients = [1., alpha, -beta]
        for i, coef1 in enumerate(coefficients):
            for j, coef2 in enumerate(coefficients):
                Cmu += (coef1 * coef2) * self.C_eta[i::3, j::3]

        sigma = np.loadtxt('./sigma_mu.txt')
        sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
        Cmu[np.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2
        return Cmu

    def compute_A_matrix(self, alpha, beta):
        """Calcula la matriz A con parámetros alpha y beta."""
        I = np.identity(740)
        a_vec = np.array([1, alpha, -beta])
        return np.tensordot(I, a_vec, axes=0).reshape((740, 2220))

    def compute_luminosity_distance(self, z, Om):
        """Calcula la distancia de luminosidad en Mpc."""
        cosmo = FlatLambdaCDM(H0=70, Om0=Om, Tcmb0=2.725)
        return cosmo.luminosity_distance(z).to_value(u.Mpc)

    def logp(self, M_B, dM_B, alpha, beta, Om):
        """Calcula el logaritmo de la probabilidad de la likelihood."""
        A = self.compute_A_matrix(alpha, beta)
        Cmu = self.compute_mu_cov(alpha, beta)
        mu_model = 5 * np.log10(self.compute_luminosity_distance(self.z, Om)) + 25
        data_model = A @ self.eta - (M_B * np.ones(740) + dM_B * self.lumvec)
        residual = data_model - mu_model
        chi2 = residual.T @ np.linalg.inv(Cmu) @ residual

        return -0.5 * chi2

