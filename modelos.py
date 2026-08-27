import numpy as np
import math
from scipy.stats import norm
from metodos_old import resolver_integral_fourier_heston, simular_monte_carlo_heston
from datos_mercado_old import get_risk_free_rate

class ModelosValoracion:
    def __init__(self,r):
        self.r = r

    # --- BLACK-SCHOLES ---
    def precio_black_scholes(self, S, K, T, sigma):
        if sigma <= 0 or T <= 0: return 0.0
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)

    # --- MERTON JUMP-DIFFUSION ---
    def precio_merton(self, S, K, T, sigma, lambd, mu_j, sigma_j, n_terminos=20):
        if sigma <= 0 or lambd < 0 or T <= 0: return 0.0
        kappa_j = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
        r_mod = self.r - lambd * kappa_j
        precio_total = 0.0

        for n in range(n_terminos):
            prob_n = (np.exp(-lambd * T) * (lambd * T) ** n) / math.factorial(n)
            sigma_n = np.sqrt(sigma ** 2 + n * (sigma_j ** 2) / T)
            r_n = r_mod + n * np.log(1 + kappa_j) / T

            d1 = (np.log(S / K) + (r_n + 0.5 * sigma_n ** 2) * T) / (sigma_n * np.sqrt(T))
            d2 = d1 - sigma_n * np.sqrt(T)
            precio_bs_n = S * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2)
            precio_total += prob_n * precio_bs_n

        return max(0.0, precio_total)

    # --- HESTON MODEL (ANALÍTICO POR FOURIER) ---
    def precio_heston(self, S, K, T, kappa, theta, xi, rho, v0):
        if T <= 0 or S <= 0 or K <= 0 or v0 <= 0: return 0.0
        P1, P2 = resolver_integral_fourier_heston(S, K, T, self.r, kappa, theta, xi, rho, v0)
        return max(0.0, S * P1 - K * np.exp(-self.r * T) * P2)

    # --- HESTON MODEL (SIMULACIÓN DE MONTE CARLO) ---
    def precio_heston_monte_carlo(self, S, K, T, kappa, theta, xi, rho, v0, num_sim, num_pasos):
        if T <= 0 or S <= 0 or K <= 0 or v0 <= 0: return 0.0
        return simular_monte_carlo_heston(S, K, T, self.r, kappa, theta, xi, rho, v0, num_sim, num_pasos)



