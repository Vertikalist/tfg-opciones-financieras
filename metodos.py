import numpy as np
import scipy.integrate as integrate


# METODOLOGÍA MÁTEMÁTICA 1: INTEGRACIÓN NUMÉRICA (FOURIER HESTON)
def char_func_heston(u, S, T, r, kappa, theta, xi, rho, v0, j):
    b = (kappa - rho * xi) if j == 1 else kappa
    f = 0.5 if j == 1 else -0.5
    i = 1j
    d = np.sqrt((rho * xi * u * i - b) ** 2 - (xi ** 2) * (2 * f * u * i - u ** 2))
    g = (b - rho * xi * u * i + d) / (b - rho * xi * u * i - d)

    D = ((b - rho * xi * u * i + d) / (xi ** 2)) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    C = r * u * i * T + (kappa * theta / (xi ** 2)) * ((b - rho * xi * u * i + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    return np.exp(C + D * v0 + i * u * np.log(S))


def integrando_heston(u, S, K, T, r, kappa, theta, xi, rho, v0, j):
    if u == 0: return 0.0
    i = 1j
    phi = char_func_heston(u, S, T, r, kappa, theta, xi, rho, v0, j)
    return np.real(np.exp(-i * u * np.log(K)) * phi / (i * u))


def resolver_integral_fourier_heston(S, K, T, r, kappa, theta, xi, rho, v0):
    P1_res, _ = integrate.quad(integrando_heston, 0, 50, args=(S, K, T, r, kappa, theta, xi, rho, v0, 1), limit=100)
    P2_res, _ = integrate.quad(integrando_heston, 0, 50, args=(S, K, T, r, kappa, theta, xi, rho, v0, 2), limit=100)

    P1 = 0.5 + P1_res / np.pi
    P2 = 0.5 + P2_res / np.pi
    return P1, P2


# METODOLOGÍA MATEMÁTICA 2: SIMULACIÓN ESTOCÁSTICA DE MONTE CARLO (HESTON)
def simular_monte_carlo_heston(S, K, T, r, kappa, theta, xi, rho, v0, num_sim, num_pasos):
    dt = T / num_pasos

    S_tray = np.zeros((num_sim, num_pasos + 1))
    v_tray = np.zeros((num_sim, num_pasos + 1))
    S_tray[:, 0] = S
    v_tray[:, 0] = v0

    np.random.seed(42)
    Z1 = np.random.normal(0.0, 1.0, (num_sim, num_pasos))
    Z2 = np.random.normal(0.0, 1.0, (num_sim, num_pasos))

    dW1 = Z1
    dW2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2

    for t in range(num_pasos):
        v_prev = v_tray[:, t]
        v_positivo = np.maximum(v_prev, 0.0)  # Full Truncation Scheme (Lord et al., 2010)

        S_tray[:, t + 1] = S_tray[:, t] * np.exp(
            (r - 0.5 * v_positivo) * dt + np.sqrt(v_positivo) * dW1[:, t] * np.sqrt(dt)
        )
        v_tray[:, t + 1] = v_prev + kappa * (theta - v_positivo) * dt + xi * np.sqrt(v_positivo) * dW2[:, t] * np.sqrt(
            dt)

    payoffs = np.maximum(S_tray[:, -1] - K, 0.0)
    return np.exp(-r * T) * np.mean(payoffs)

