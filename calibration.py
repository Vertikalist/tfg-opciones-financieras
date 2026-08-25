import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from config import BOUNDS_MERTON, BOUNDS_HESTON, SEMILLA_ALEATORIA, PENALIZACION_FELLER, N_RESTARTS


# ==============================================================================
# FUNCIONES DE PÉRDIDA INTERNAS (LOSS FUNCTIONS)
# ==============================================================================
def loss_black_scholes(params, S, T, df, calc_model):
    sigma = params
    errores = [(fila['C_mkt'] - calc_model.precio_black_scholes(S, fila['K'], T, sigma)) ** 2 for _, fila in
               df.iterrows()]
    return np.mean(errores)


def loss_merton(params, S, T, df, calc_model):
    sigma, lambd, mu_j, sigma_j = params
    errores = [(fila['C_mkt'] - calc_model.precio_merton(S, fila['K'], T, sigma, lambd, mu_j, sigma_j)) ** 2 for _, fila
               in df.iterrows()]
    return np.mean(errores)


def loss_heston(params, S, T, df, calc_model):
    kappa, theta, xi, rho, v0 = params
    penalizacion = 0.0
    if 2 * kappa * theta <= xi ** 2:
        penalizacion = PENALIZACION_FELLER * (xi ** 2 - 2 * kappa * theta)

    errores = [(fila['C_mkt'] - calc_model.precio_heston(S, fila['K'], T, kappa, theta, xi, rho, v0)) ** 2 for _, fila
               in df.iterrows()]
    return np.mean(errores) + penalizacion


# ==============================================================================
# ENGINES DE CALIBRACIÓN COMPLETA (PRICE-RMSE)
# ==============================================================================
def calibrar_black_scholes(S, T, df, calc_model):
    bounds = [(0.001, 2.0)]
    res = minimize(loss_black_scholes, 0.20, args=(S, T, df, calc_model), bounds=bounds, method='L-BFGS-B')
    return res


def calibrar_merton(S, T, df, calc_model):
    rng = np.random.default_rng(SEMILLA_ALEATORIA)
    mejor = None
    for _ in range(N_RESTARTS):
        x0 = [rng.uniform(lo, hi) for lo, hi in BOUNDS_MERTON]
        res = minimize(loss_merton, x0, args=(S, T, df, calc_model), bounds=BOUNDS_MERTON, method='L-BFGS-B')
        if mejor is None or res.fun < mejor.fun:
            mejor = res
    return mejor


def calibrar_heston(S, T, df, calc_model):
    rng = np.random.default_rng(SEMILLA_ALEATORIA)
    mejor = None
    for _ in range(N_RESTARTS):
        x0 = [rng.uniform(lo, hi) for lo, hi in BOUNDS_HESTON]
        res = minimize(loss_heston, x0, args=(S, T, df, calc_model), bounds=BOUNDS_HESTON, method='L-BFGS-B')
        if mejor is None or res.fun < mejor.fun:
            mejor = res
    return mejor


# ==============================================================================
# CALIBRACIÓN POR VOLATILIDAD IMPLÍCITA (IV-RMSE)
# ==============================================================================

def _precio_bs_puro(S, K, T, r, sigma):
    """Black-Scholes call price (función auxiliar interna para inversión de IV)."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_bs(precio, S, K, T, r, sigma0=0.2, max_iter=100, tol=1e-8):
    """
    Calcula la volatilidad implícita de Black-Scholes mediante Newton-Raphson.

    La iteración es: sigma_{n+1} = sigma_n - (BS(sigma_n) - C_mkt) / vega(sigma_n)
    donde vega = S * sqrt(T) * N'(d1).

    Devuelve np.nan si el precio está fuera del rango arbitrable o si el metodo
    no converge (la fila se ignora en los cálculos de IV-RMSE).
    """
    precio_min = max(0.0, S - K * np.exp(-r * T))  # Cota inferior (paridad put-call)
    precio_max = S                                   # Cota superior (call <= S)

    if precio <= precio_min or precio >= precio_max:
        return np.nan

    sigma = sigma0
    for _ in range(max_iter):
        if sigma <= 0 or T <= 0:
            return np.nan

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        precio_modelo = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * np.sqrt(T) * norm.pdf(d1)

        if abs(vega) < 1e-10:   # Vega casi nula: convergencia numérica inestable
            return np.nan

        diff = precio_modelo - precio
        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega

    return np.nan  # No convergió en max_iter iteraciones


# --- Funciones de pérdida IV-RMSE ---
# IV_mkt se pasa precalculada (array) para no recomputarla en cada evaluación del optimizador.

def loss_iv_black_scholes(params, S, T, iv_mkt_arr, Ks, calc_model):
    sigma = params
    errores = []
    for iv_mkt, K in zip(iv_mkt_arr, Ks):
        if np.isnan(iv_mkt):
            continue
        precio_modelo = calc_model.precio_black_scholes(S, K, T, sigma)
        iv_modelo = implied_vol_bs(precio_modelo, S, K, T, calc_model.r)
        if np.isnan(iv_modelo):
            continue
        errores.append((iv_mkt - iv_modelo) ** 2)
    return np.mean(errores) if errores else 1e6


def loss_iv_merton(params, S, T, iv_mkt_arr, Ks, calc_model):
    sigma, lambd, mu_j, sigma_j = params
    errores = []
    for iv_mkt, K in zip(iv_mkt_arr, Ks):
        if np.isnan(iv_mkt):
            continue
        precio_modelo = calc_model.precio_merton(S, K, T, sigma, lambd, mu_j, sigma_j)
        iv_modelo = implied_vol_bs(precio_modelo, S, K, T, calc_model.r)
        if np.isnan(iv_modelo):
            continue
        errores.append((iv_mkt - iv_modelo) ** 2)
    return np.mean(errores) if errores else 1e6


def loss_iv_heston(params, S, T, iv_mkt_arr, Ks, calc_model):
    kappa, theta, xi, rho, v0 = params
    penalizacion = 0.0
    if 2 * kappa * theta <= xi ** 2:
        penalizacion = PENALIZACION_FELLER * (xi ** 2 - 2 * kappa * theta)

    errores = []
    for iv_mkt, K in zip(iv_mkt_arr, Ks):
        if np.isnan(iv_mkt):
            continue
        precio_modelo = calc_model.precio_heston(S, K, T, kappa, theta, xi, rho, v0)
        iv_modelo = implied_vol_bs(precio_modelo, S, K, T, calc_model.r)
        if np.isnan(iv_modelo):
            continue
        errores.append((iv_mkt - iv_modelo) ** 2)
    return (np.mean(errores) if errores else 1e6) + penalizacion


# --- Engines de calibración IV-RMSE ---

def _precomputar_iv_mkt(df, S, T, r):
    """Precalcula las IVs de mercado una sola vez antes de la optimización."""
    Ks = df['K'].values
    iv_mkt_arr = np.array([implied_vol_bs(c, S, K, T, r) for c, K in zip(df['C_mkt'].values, Ks)])
    return iv_mkt_arr, Ks


def calibrar_black_scholes_iv(S, T, df, calc_model):
    iv_mkt_arr, Ks = _precomputar_iv_mkt(df, S, T, calc_model.r)
    bounds = [(0.001, 2.0)]
    res = minimize(loss_iv_black_scholes, 0.20, args=(S, T, iv_mkt_arr, Ks, calc_model),
                   bounds=bounds, method='L-BFGS-B')
    return res


def calibrar_merton_iv(S, T, df, calc_model):
    iv_mkt_arr, Ks = _precomputar_iv_mkt(df, S, T, calc_model.r)
    rng = np.random.default_rng(SEMILLA_ALEATORIA)
    mejor = None
    for _ in range(N_RESTARTS):
        x0 = [rng.uniform(lo, hi) for lo, hi in BOUNDS_MERTON]
        res = minimize(loss_iv_merton, x0, args=(S, T, iv_mkt_arr, Ks, calc_model),
                       bounds=BOUNDS_MERTON, method='L-BFGS-B')
        if mejor is None or res.fun < mejor.fun:
            mejor = res
    return mejor


def calibrar_heston_iv(S, T, df, calc_model):
    iv_mkt_arr, Ks = _precomputar_iv_mkt(df, S, T, calc_model.r)
    rng = np.random.default_rng(SEMILLA_ALEATORIA)
    mejor = None
    for _ in range(N_RESTARTS):
        x0 = [rng.uniform(lo, hi) for lo, hi in BOUNDS_HESTON]
        res = minimize(loss_iv_heston, x0, args=(S, T, iv_mkt_arr, Ks, calc_model),
                       bounds=BOUNDS_HESTON, method='L-BFGS-B')
        if mejor is None or res.fun < mejor.fun:
            mejor = res
    return mejor
