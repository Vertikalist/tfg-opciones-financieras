import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (necesario para projection='3d')
from scipy.optimize import minimize

from calibration import implied_vol_bs
from config import BOUNDS_MERTON, BOUNDS_HESTON, SEMILLA_ALEATORIA, PENALIZACION_FELLER
from metodos import simular_monte_carlo_heston_stats
from validacion_oos import calcular_metricas_oos


# ============================================================================
# 1. CONVERGENCIA MONTE CARLO CON INTERVALOS DE CONFIANZA
# ============================================================================

def grafico_convergencia_mc_ic(S, K, T, calc_model, params_heston, num_pasos,
                                n_min=500, n_max=50000, n_nodos=18):
    """
    Traza la convergencia del estimador Monte Carlo hacia el precio exacto de
    Fourier, añadiendo bandas de confianza al 95% (± 1.96 · error estándar).
    """
    kappa, theta, xi, rho, v0 = params_heston
    precio_fourier = calc_model.precio_heston(S, K, T, kappa, theta, xi, rho, v0)

    rangos_sim = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), n_nodos).astype(int))

    precios, ses = [], []
    print(f"-> [EXT] Convergencia MC con IC ({len(rangos_sim)} nodos, hasta {n_max} trayectorias)...")
    for sim in rangos_sim:
        p, se = simular_monte_carlo_heston_stats(
            S, K, T, calc_model.r, kappa, theta, xi, rho, v0, int(sim), num_pasos)
        precios.append(p)
        ses.append(se)

    precios = np.array(precios)
    ses = np.array(ses)
    ic = 1.96 * ses

    df = pd.DataFrame({
        'Simulaciones': rangos_sim,
        'Precio_MC': precios,
        'SE': ses,
        'IC95_inf': precios - ic,
        'IC95_sup': precios + ic,
    })

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 5.5))
    plt.fill_between(rangos_sim, precios - ic, precios + ic,
                      color='#1f77b4', alpha=0.20, label='IC 95% (± 1.96·SE)')
    plt.plot(rangos_sim, precios, color='#1f77b4', linewidth=2, marker='o',
             markersize=5, label='Precio Monte Carlo')
    plt.axhline(y=precio_fourier, color='r', linestyle='--', linewidth=2,
                label=f'Precio exacto Fourier = {precio_fourier:.2f}')
    plt.xscale('log')
    plt.title(f'Convergencia de Monte Carlo con IC 95% (Heston, K = {K:.0f})',
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Número de trayectorias (escala log)')
    plt.ylabel('Precio de la opción Call')
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='best')
    plt.tight_layout()
    plt.savefig('convergencia_mc_ic_tfg.png', dpi=300)
    plt.close()
    print("  [EXT] Guardado: convergencia_mc_ic_tfg.png")
    return df


# ============================================================================
# 2. VALIDACIÓN OOS MULTI-HORIZONTE
# ============================================================================

def analisis_oos_multihorizonte(S, calc_model, params_bs, params_merton, params_heston,
                                 horizontes):
    """
    Aplica los parámetros calibrados in-sample (T1) sobre varios vencimientos
    OOS sin recalibrar. `horizontes` es una lista de tuplas (T_oos, df_oos).
    Devuelve un DataFrame tidy y guarda un gráfico de barras comparativo.
    """
    filas = []
    for T_oos, df_oos in horizontes:
        dias = int(round(T_oos * 365))
        metricas = calcular_metricas_oos(
            df_oos, S, T_oos, calc_model, params_bs, params_merton, params_heston)
        for modelo, m in metricas.items():
            filas.append({
                'Horizonte_dias': dias,
                'T': T_oos,
                'Modelo': modelo,
                'RMSE_precio': m['RMSE_precio'],
                'IV_RMSE_pct': m['IV_RMSE'] * 100 if not np.isnan(m['IV_RMSE']) else np.nan,
            })
    df = pd.DataFrame(filas)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    sns.barplot(ax=axes[0], data=df, x='Horizonte_dias', y='RMSE_precio',
                hue='Modelo', palette='Set1')
    axes[0].set_title('RMSE de precio OOS por horizonte', fontweight='bold')
    axes[0].set_xlabel('Horizonte OOS (días)')
    axes[0].set_ylabel('RMSE de precio')

    sns.barplot(ax=axes[1], data=df, x='Horizonte_dias', y='IV_RMSE_pct',
                hue='Modelo', palette='Set1')
    axes[1].set_title('IV-RMSE OOS por horizonte', fontweight='bold')
    axes[1].set_xlabel('Horizonte OOS (días)')
    axes[1].set_ylabel('IV-RMSE (%)')

    plt.savefig('oos_multihorizonte_tfg.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [EXT] Guardado: oos_multihorizonte_tfg.png")
    return df


# ============================================================================
# 3. CALIBRACIÓN DE LA SUPERFICIE DE VOLATILIDAD (JOINT / MULTI-VENCIMIENTO)
# ============================================================================

def _loss_bs_superficie(params, S, superficie, calc):
    sigma = params[0]
    err = []
    for T, df in superficie:
        for _, f in df.iterrows():
            err.append((f['C_mkt'] - calc.precio_black_scholes(S, f['K'], T, sigma)) ** 2)
    return np.mean(err)


def _loss_merton_superficie(params, S, superficie, calc):
    sigma, lambd, mu_j, sigma_j = params
    err = []
    for T, df in superficie:
        for _, f in df.iterrows():
            err.append((f['C_mkt'] - calc.precio_merton(S, f['K'], T, sigma, lambd, mu_j, sigma_j)) ** 2)
    return np.mean(err)


def _loss_heston_superficie(params, S, superficie, calc):
    kappa, theta, xi, rho, v0 = params
    pen = 0.0
    if 2 * kappa * theta <= xi ** 2:
        pen = PENALIZACION_FELLER * (xi ** 2 - 2 * kappa * theta)
    err = []
    for T, df in superficie:
        for _, f in df.iterrows():
            err.append((f['C_mkt'] - calc.precio_heston(S, f['K'], T, kappa, theta, xi, rho, v0)) ** 2)
    return np.mean(err) + pen


def calibrar_superficie(S, superficie, calc, restarts_heston=2, restarts_merton=3,
                         submuestreo=2):
    """
    Calibra los tres modelos CONJUNTAMENTE sobre todos los vencimientos.
    Para acelerar, la calibración usa un submuestreo de strikes (1 de cada
    `submuestreo`); las métricas y gráficos posteriores usan la superficie completa.

    Devuelve un dict con los parámetros calibrados y el MSE global de cada modelo.
    """
    superficie_cal = [(T, df.iloc[::submuestreo].reset_index(drop=True)) for T, df in superficie]
    rng = np.random.default_rng(SEMILLA_ALEATORIA)

    # --- Black-Scholes (1 sigma para toda la superficie) ---
    print("-> [SURF] Calibrando Black-Scholes (superficie)...")
    res_bs = minimize(_loss_bs_superficie, [0.20], args=(S, superficie_cal, calc),
                       bounds=[(0.001, 2.0)], method='L-BFGS-B')

    # --- Merton ---
    print("-> [SURF] Calibrando Merton (superficie)...")
    mejor_m = None
    for _ in range(restarts_merton):
        x0 = [rng.uniform(lo, hi) for lo, hi in BOUNDS_MERTON]
        r = minimize(_loss_merton_superficie, x0, args=(S, superficie_cal, calc),
                     bounds=BOUNDS_MERTON, method='L-BFGS-B')
        if mejor_m is None or r.fun < mejor_m.fun:
            mejor_m = r

    # --- Heston ---
    print("-> [SURF] Calibrando Heston (superficie)...")
    mejor_h = None
    for _ in range(restarts_heston):
        x0 = [rng.uniform(lo, hi) for lo, hi in BOUNDS_HESTON]
        r = minimize(_loss_heston_superficie, x0, args=(S, superficie_cal, calc),
                     bounds=BOUNDS_HESTON, method='L-BFGS-B')
        if mejor_h is None or r.fun < mejor_h.fun:
            mejor_h = r

    return {
        'sigma_bs': float(res_bs.x[0]),
        'merton': tuple(mejor_m.x),
        'heston': tuple(mejor_h.x),
        'mse_bs': float(res_bs.fun),
        'mse_merton': float(mejor_m.fun),
        'mse_heston': float(mejor_h.fun),
    }


def metricas_superficie(S, superficie, calc, params):
    """Calcula RMSE de precio e IV-RMSE por vencimiento para la superficie calibrada."""
    r = calc.r
    filas = []
    for T, df in superficie:
        dias = int(round(T * 365))
        for modelo, precio_fn in _modelos_precio(S, T, calc, params).items():
            err_p, err_iv = [], []
            for _, f in df.iterrows():
                pm = precio_fn(f['K'])
                err_p.append((f['C_mkt'] - pm) ** 2)
                iv_mkt = implied_vol_bs(f['C_mkt'], S, f['K'], T, r)
                iv_mod = implied_vol_bs(pm, S, f['K'], T, r)
                if not np.isnan(iv_mkt) and not np.isnan(iv_mod):
                    err_iv.append((iv_mkt - iv_mod) ** 2)
            filas.append({
                'Vencimiento_dias': dias,
                'Modelo': modelo,
                'RMSE_precio': np.sqrt(np.mean(err_p)),
                'IV_RMSE_pct': (np.sqrt(np.mean(err_iv)) * 100) if err_iv else np.nan,
            })
    return pd.DataFrame(filas)


def _modelos_precio(S, T, calc, params):
    return {
        'Black-Scholes': lambda K: calc.precio_black_scholes(S, K, T, params['sigma_bs']),
        'Merton':        lambda K: calc.precio_merton(S, K, T, *params['merton']),
        'Heston':        lambda K: calc.precio_heston(S, K, T, *params['heston']),
    }


# ============================================================================
# 4. GRÁFICOS DE SUPERFICIE
# ============================================================================

def _iv_serie(precios, S, Ks, T, r):
    return np.array([implied_vol_bs(p, S, k, T, r) for p, k in zip(precios, Ks)])


def graficar_smiles_superficie(S, superficie, calc, params):
    """Rejilla de smiles: una subgráfica por vencimiento (mercado vs modelos)."""
    r = calc.r
    n = len(superficie)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    colores = {'Black-Scholes': '#e41a1c', 'Merton': '#377eb8', 'Heston': '#4daf4a'}

    for idx, (T, df) in enumerate(superficie):
        ax = axes[idx]
        Ks = df['K'].values
        iv_mkt = _iv_serie(df['C_mkt'].values, S, Ks, T, r)
        ax.scatter(Ks, iv_mkt * 100, color='black', marker='X', s=45,
                   label='Mercado', zorder=5)
        for modelo, fn in _modelos_precio(S, T, calc, params).items():
            precios = np.array([fn(k) for k in Ks])
            iv = _iv_serie(precios, S, Ks, T, r)
            ax.plot(Ks, iv * 100, color=colores[modelo], linewidth=2, label=modelo)
        ax.axvline(x=S, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f'T = {int(round(T*365))} días', fontweight='bold')
        ax.set_xlabel('Strike (K)')
        ax.set_ylabel('IV (%)')
        if idx == 0:
            ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Sonrisa de volatilidad por vencimiento (superficie calibrada)',
                 fontsize=14, fontweight='bold')
    plt.savefig('superficie_smiles_tfg.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [EXT] Guardado: superficie_smiles_tfg.png")


def graficar_superficie_3d(S, superficie, calc, params):
    """Superficie 3D: IV de mercado vs IV de Heston sobre (Strike, Vencimiento)."""
    r = calc.r
    Ks_grid, Ts_grid, iv_mkt_grid, iv_hes_grid = [], [], [], []
    for T, df in superficie:
        Ks = df['K'].values
        iv_mkt = _iv_serie(df['C_mkt'].values, S, Ks, T, r)
        precios_h = np.array([calc.precio_heston(S, k, T, *params['heston']) for k in Ks])
        iv_h = _iv_serie(precios_h, S, Ks, T, r)
        for k, ivm, ivh in zip(Ks, iv_mkt, iv_h):
            if not np.isnan(ivm) and not np.isnan(ivh):
                Ks_grid.append(k)
                Ts_grid.append(T * 365)
                iv_mkt_grid.append(ivm * 100)
                iv_hes_grid.append(ivh * 100)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Ks_grid, Ts_grid, iv_mkt_grid, cmap='viridis', alpha=0.75,
                     edgecolor='none')
    ax.scatter(Ks_grid, Ts_grid, iv_hes_grid, color='red', s=12,
               label='IV Heston (calibrado)')
    ax.set_xlabel('Strike (K)')
    ax.set_ylabel('Vencimiento (días)')
    ax.set_zlabel('IV (%)')
    ax.set_title('Superficie de volatilidad: Mercado (malla) vs Heston (puntos)',
                  fontweight='bold')
    ax.legend()
    ax.view_init(elev=22, azim=-60)
    plt.savefig('superficie_3d_tfg.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [EXT] Guardado: superficie_3d_tfg.png")


def graficar_term_structure_atm(S, superficie, calc, params):
    """Estructura temporal de la IV ATM: mercado vs modelos por vencimiento."""
    r = calc.r
    filas = []
    for T, df in superficie:
        idx_atm = (df['K'] - S).abs().idxmin()
        K_atm = df.loc[idx_atm, 'K']
        iv_mkt = implied_vol_bs(df.loc[idx_atm, 'C_mkt'], S, K_atm, T, r)
        fila = {'dias': int(round(T * 365)), 'Mercado': iv_mkt}
        for modelo, fn in _modelos_precio(S, T, calc, params).items():
            fila[modelo] = implied_vol_bs(fn(K_atm), S, K_atm, T, r)
        filas.append(fila)
    df_ts = pd.DataFrame(filas)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.figure(figsize=(10, 5.5))
    plt.plot(df_ts['dias'], df_ts['Mercado'] * 100, 'kX--', markersize=11,
             linewidth=1.5, label='Mercado')
    for modelo, color in [('Black-Scholes', '#e41a1c'), ('Merton', '#377eb8'),
                           ('Heston', '#4daf4a')]:
        plt.plot(df_ts['dias'], df_ts[modelo] * 100, marker='o', color=color,
                 linewidth=2, label=modelo)
    plt.title('Estructura temporal de la volatilidad ATM', fontweight='bold')
    plt.xlabel('Vencimiento (días)')
    plt.ylabel('IV ATM (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('term_structure_atm_tfg.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [EXT] Guardado: term_structure_atm_tfg.png")
    return df_ts

