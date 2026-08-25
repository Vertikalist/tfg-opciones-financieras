"""
Validación Estática Out-of-Sample (OOS)
========================================
Estrategia: calibrar sobre el vencimiento T1 (in-sample) y aplicar
los parámetros obtenidos, sin reoptimizar, sobre el vencimiento T2
(siguiente expiry disponible). Como el precio spot S es el mismo en
ambos casos (misma sesión de mercado), esto equivale a testear si el
modelo generaliza a un horizonte temporal distinto al calibrado.

Uso desde main.py (añadir al final, tras las calibraciones):
    from validacion_oos import ejecutar_validacion_oos
    ejecutar_validacion_oos(S_mercado, T_mercado, calc,
                            res_bs, res_merton, res_heston)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from calibration import implied_vol_bs
from config import TICKER_ACTIVO, MIN_DIAS_VENCIMIENTO, MIN_OPCIONES_LIQUIDAS


# ==============================================================================
# OBTENCIÓN DE DATOS OOS (siguiente expiry)
# ==============================================================================

def obtener_datos_siguiente_expiry(S, T_insample, ticker_symbol=TICKER_ACTIVO):
    """
    Descarga la cadena de opciones del primer vencimiento POSTERIOR a T_insample
    que tenga al menos MIN_OPCIONES_LIQUIDAS opciones líquidas en el rango ±20% de S.

    Devuelve (T_oos, df_oos) o lanza ValueError si no se encuentra ninguno.
    """
    ticker = yf.Ticker(ticker_symbol)
    vencimientos = ticker.options
    fecha_hoy = pd.Timestamp.today()

    # Convertir T_insample (años) a días aproximados para comparar
    dias_insample = T_insample * 365

    for fecha in vencimientos:
        dias = (pd.to_datetime(fecha) - fecha_hoy).days
        # Queremos el siguiente expiry: más días que el in-sample y >= MIN_DIAS
        if dias <= dias_insample + 1:
            continue
        if dias < MIN_DIAS_VENCIMIENTO:
            continue

        T_candidato = dias / 365.0
        calls = ticker.option_chain(fecha).calls
        calls['Mid_Price'] = (calls['bid'] + calls['ask']) / 2

        datos = calls[
            (calls['bid'] > 0) &
            (calls['openInterest'] > 0) &
            (calls['strike'] > 0.8 * S) &
            (calls['strike'] < 1.2 * S)
        ].copy()

        if len(datos) >= MIN_OPCIONES_LIQUIDAS:
            df_oos = datos[['strike', 'Mid_Price']].rename(
                columns={'strike': 'K', 'Mid_Price': 'C_mkt'}
            )
            print(f"   [OOS] Expiry OOS: {fecha} | T_oos = {T_candidato:.4f} años | {len(df_oos)} opciones")
            return T_candidato, df_oos

    raise ValueError(
        f"No se encontró ningún vencimiento OOS válido posterior a T={T_insample:.4f}. "
        f"Vencimientos disponibles: {list(vencimientos)}"
    )


# ==============================================================================
# MÉTRICAS OOS
# ==============================================================================

def calcular_metricas_oos(df, S, T, calc_model, params_bs, params_merton, params_heston):
    """
    Aplica los parámetros calibrados en T1 a los precios de mercado de T2.
    Devuelve RMSE de precio e IV-RMSE para cada modelo.
    """
    sigma_bs = float(np.asarray(params_bs).squeeze())
    r = calc_model.r

    modelos = {
        'Black-Scholes': lambda K: calc_model.precio_black_scholes(S, K, T, sigma_bs),
        'Merton':        lambda K: calc_model.precio_merton(S, K, T, *params_merton),
        'Heston':        lambda K: calc_model.precio_heston(S, K, T, *params_heston),
    }

    resultados = {}
    for nombre, precio_fn in modelos.items():
        errores_precio, errores_iv = [], []
        for _, fila in df.iterrows():
            precio_modelo = precio_fn(fila['K'])
            errores_precio.append((fila['C_mkt'] - precio_modelo) ** 2)

            iv_mkt    = implied_vol_bs(fila['C_mkt'],    S, fila['K'], T, r)
            iv_modelo = implied_vol_bs(precio_modelo, S, fila['K'], T, r)
            if not np.isnan(iv_mkt) and not np.isnan(iv_modelo):
                errores_iv.append((iv_mkt - iv_modelo) ** 2)

        resultados[nombre] = {
            'RMSE_precio': np.sqrt(np.mean(errores_precio)),
            'IV_RMSE':     np.sqrt(np.mean(errores_iv)) if errores_iv else np.nan,
        }
    return resultados


# ==============================================================================
# GRÁFICOS OOS (mismo estilo que generar_graficos_comparativos)
# ==============================================================================

def graficar_validacion_oos(df, S, T_insample, T_oos, calc_model,
                             params_bs, params_merton, params_heston):
    """
    Dos paneles:
      - Izquierdo: precios OOS (T2) vs modelos calibrados en T1
      - Derecho:   residuos por strike
    """
    sigma_bs = float(np.asarray(params_bs).squeeze())

    df_plot = df.copy()
    df_plot['Black-Scholes'] = df_plot['K'].apply(
        lambda k: calc_model.precio_black_scholes(S, k, T_oos, sigma_bs))
    df_plot['Merton'] = df_plot['K'].apply(
        lambda k: calc_model.precio_merton(S, k, T_oos, *params_merton))
    df_plot['Heston'] = df_plot['K'].apply(
        lambda k: calc_model.precio_heston(S, k, T_oos, *params_heston))

    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Panel 1: precios
    df_long = df_plot.melt(
        id_vars=['K', 'C_mkt'],
        value_vars=['Black-Scholes', 'Merton', 'Heston'],
        var_name='Modelo', value_name='Precio_Modelo'
    ).dropna(subset=['Precio_Modelo'])

    sns.scatterplot(ax=axes[0], data=df_plot, x='K', y='C_mkt',
                    color='black', marker='X', s=100, label='Mercado (OOS)', zorder=5)
    sns.lineplot(ax=axes[0], data=df_long, x='K', y='Precio_Modelo',
                 hue='Modelo', style='Modelo', palette='Set1', linewidth=2.5)
    axes[0].axvline(x=S, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[0].annotate('ATM', xy=(S, axes[0].get_ylim()[1] * 0.92),
                     ha='center', color='gray', fontsize=10)
    axes[0].set_title(
        f'OOS – Precios  (calibrado: T={T_insample:.3f}a  |  validado: T={T_oos:.3f}a)',
        fontweight='bold'
    )
    axes[0].set_xlabel('Strike (K)')
    axes[0].set_ylabel('Precio Call')
    axes[0].legend(title='')

    # Panel 2: residuos
    df_err = pd.DataFrame({'K': df_plot['K']})
    for m in ['Black-Scholes', 'Merton', 'Heston']:
        df_err[m] = df_plot['C_mkt'] - df_plot[m]

    df_err_long = df_err.melt(
        id_vars=['K'], value_vars=['Black-Scholes', 'Merton', 'Heston'],
        var_name='Modelo', value_name='Error'
    ).dropna(subset=['Error'])

    sns.lineplot(ax=axes[1], data=df_err_long, x='K', y='Error',
                 hue='Modelo', style='Modelo', marker='o', markersize=8,
                 palette='Set1', linewidth=2.5)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].set_title('Residuos OOS por Strike', fontweight='bold')
    axes[1].set_xlabel('Strike (K)')
    axes[1].set_ylabel('Error (Mercado OOS – Modelo)')
    axes[1].legend(title='')

    plt.savefig('validacion_oos_tfg.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(" [OOS] Gráfico guardado como 'validacion_oos_tfg.png'\n")


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

def ejecutar_validacion_oos(S, T_insample, calc_model, res_bs, res_merton, res_heston,
                             ticker_symbol=TICKER_ACTIVO):
    """
    Orquesta la validación OOS en un solo paso:
      1. Descarga opciones del siguiente expiry disponible (T2 > T1)
      2. Aplica los parámetros de T1 sin reoptimizar
      3. Imprime tabla comparativa RMSE / IV-RMSE
      4. Genera gráfico de precios y residuos

    Llamar desde main.py tras las calibraciones:
        ejecutar_validacion_oos(S_mercado, T_mercado, calc,
                                res_bs, res_merton, res_heston)
    """
    print("\n" + "=" * 60)
    print("VALIDACIÓN ESTÁTICA OUT-OF-SAMPLE")
    print("=" * 60)

    try:
        T_oos, df_oos = obtener_datos_siguiente_expiry(S, T_insample, ticker_symbol)
    except ValueError as e:
        print(f"   [OOS] No se pudo obtener datos OOS: {e}")
        return None

    params_bs     = res_bs.x
    params_merton = res_merton.x
    params_heston = res_heston.x

    print(f"   In-sample:  T1 = {T_insample:.4f} años")
    print(f"   Out-sample: T2 = {T_oos:.4f} años  |  {len(df_oos)} opciones\n")

    metricas = calcular_metricas_oos(
        df_oos, S, T_oos, calc_model,
        params_bs, params_merton, params_heston
    )

    tabla = pd.DataFrame([
        {
            'Modelo':            nombre,
            'RMSE Precio (OOS)': f"{m['RMSE_precio']:.4f}",
            'IV-RMSE (OOS)':     f"{m['IV_RMSE']*100:.4f}%" if not np.isnan(m['IV_RMSE']) else "N/A",
        }
        for nombre, m in metricas.items()
    ])
    print(tabla.to_string(index=False))
    print("=" * 60)

    graficar_validacion_oos(
        df_oos, S, T_insample, T_oos, calc_model,
        params_bs, params_merton, params_heston
    )

    return metricas
