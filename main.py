from datetime import date

import numpy as np
import pandas as pd

from config import NUM_SIMULACIONES, NUM_PASOS_TIEMPO, TICKER_ACTIVO import datos_mercado as dme
import analisis as ext


def _sep(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def run(ticker_symbol=TICKER_ACTIVO):
    # ----------------------------------------------------------------
    # 1. Descarga de datos reales (varios vencimientos)
    # ----------------------------------------------------------------
    _sep(f"DESCARGA DE DATOS REALES (Yahoo Finance) – {ticker_symbol}")
    tasa_r = float(get_risk_free_rate("1mo"))
    S_spot, q, cadenas = dme.obtener_cadenas(ticker_symbol)

    # Precio spot ajustado por dividendos al vencimiento in-sample; se reutiliza
    # como spot de referencia en OOS y superficie (el efecto de q entre
    # vencimientos es de segundo orden).
    (T1, df1), horizontes = dme.separar_insample_oos(cadenas)
    S = S_spot * np.exp(-q * T1)
    calc = ModelosValoracion(r=tasa_r)

    # ----------------------------------------------------------------
    # 2. Calibración in-sample
    # ----------------------------------------------------------------
    _sep("CALIBRACIÓN IN-SAMPLE")
    print(f"r = {tasa_r:.4f} | S_adj = {S:.2f} | T1 = {T1:.4f} años | {len(df1)} strikes\n")
    print("-> Calibrando Black-Scholes...")
    res_bs = calibrar_black_scholes(S, T1, df1, calc)
    print("-> Calibrando Merton...")
    res_merton = calibrar_merton(S, T1, df1, calc)
    print("-> Calibrando Heston...")
    res_heston = calibrar_heston(S, T1, df1, calc)

    tabla_is = pd.DataFrame({
        'Modelo': ['Black-Scholes', 'Merton', 'Heston'],
        'MSE Final': [res_bs.fun, res_merton.fun, res_heston.fun],
        'Parámetros': [1, 4, 5],
    })
    print("\n" + tabla_is.to_string(index=False))

    kap, the, xi, rho, v0 = res_heston.x
    print("\nParámetros Heston (in-sample, 1 vencimiento):")
    print(f"  kappa={kap:.4f}  theta={the:.4f}  xi={xi:.4f}  rho={rho:.4f}  v0={v0:.4f}")

    # ----------------------------------------------------------------
    # 3. Gráficos base
    # ----------------------------------------------------------------
    _sep("GRÁFICOS COMPARATIVOS BASE")
    generar_graficos_comparativos(df1, S, T1, calc, res_bs.x, res_merton.x, res_heston.x)
    generar_graficos_comparativos_iv(df1, S, T1, calc, res_bs.x, res_merton.x, res_heston.x)

    # ----------------------------------------------------------------
    # 4. Fourier vs Monte Carlo con IC
    # ----------------------------------------------------------------
    _sep("VALIDACIÓN FOURIER vs MONTE CARLO (con IC 95%)")
    K_atm = float(df1.iloc[(df1['K'] - S).abs().idxmin()]['K'])
    precio_fourier = calc.precio_heston(S, K_atm, T1, kap, the, xi, rho, v0)
    precio_mc, se_mc = ext.simular_monte_carlo_heston_stats(
        S, K_atm, T1, calc.r, kap, the, xi, rho, v0, NUM_SIMULACIONES, NUM_PASOS_TIEMPO)
    print(f"K_ATM = {K_atm:.0f}")
    print(f"  - Fourier:     {precio_fourier:.4f}")
    print(f"  - Monte Carlo: {precio_mc:.4f}  (SE = {se_mc:.4f}, IC95 ± {1.96*se_mc:.4f})")
    print(f"  - Error abs.:  {abs(precio_fourier - precio_mc):.4f}")
    ext.grafico_convergencia_mc_ic(S, K_atm, T1, calc, res_heston.x, NUM_PASOS_TIEMPO)

    # ----------------------------------------------------------------
    # 5. OOS multi-horizonte
    # ----------------------------------------------------------------
    _sep("VALIDACIÓN OOS MULTI-HORIZONTE")
    if horizontes:
        df_oos = ext.analisis_oos_multihorizonte(
            S, calc, res_bs.x, res_merton.x, res_heston.x, horizontes)
        print(df_oos.to_string(index=False))
    else:
        print("No hay vencimientos adicionales para OOS.")
        df_oos = pd.DataFrame(columns=['Horizonte_dias', 'Modelo', 'RMSE_precio', 'IV_RMSE_pct'])

    # ----------------------------------------------------------------
    # 6. Superficie de volatilidad
    # ----------------------------------------------------------------
    _sep("CALIBRACIÓN DE SUPERFICIE (multi-vencimiento)")
    superficie = dme.construir_superficie(cadenas)
    params_surf = ext.calibrar_superficie(S, superficie, calc)
    print("\nParámetros de superficie calibrados:")
    print(f"  BS sigma = {params_surf['sigma_bs']:.4f}  (MSE {params_surf['mse_bs']:.4f})")
    print(f"  Merton   = {tuple(round(x,4) for x in params_surf['merton'])}  (MSE {params_surf['mse_merton']:.4f})")
    print(f"  Heston   = {tuple(round(x,4) for x in params_surf['heston'])}  (MSE {params_surf['mse_heston']:.4f})")

    df_surf_met = ext.metricas_superficie(S, superficie, calc, params_surf)
    print("\nMétricas de superficie por vencimiento:")
    print(df_surf_met.to_string(index=False))

    ext.graficar_smiles_superficie(S, superficie, calc, params_surf)
    ext.graficar_superficie_3d(S, superficie, calc, params_surf)
    df_ts = ext.graficar_term_structure_atm(S, superficie, calc, params_surf)
    print("\nEstructura temporal de IV ATM:")
    print(df_ts.to_string(index=False))

    # ----------------------------------------------------------------
    # 7. Guardar resultados para los exportadores (Word/LaTeX)
    # ----------------------------------------------------------------
    meta = {"ticker": ticker_symbol, "fecha": str(date.today()),
            "S": float(S), "r": float(tasa_r), "T1": float(T1), "n_strikes": int(len(df1))}
    fourier_mc = {"K_atm": float(K_atm), "fourier": float(precio_fourier),
                  "mc": float(precio_mc), "se": float(se_mc),
                  "ic95": float(1.96 * se_mc), "error": float(abs(precio_fourier - precio_mc))}
    resultados = contenido_datos.construir_resultados(
        meta, res_bs, res_merton, res_heston, fourier_mc, df_oos, params_surf, df_surf_met)
    contenido_datos.guardar(resultados)

    _sep("FIN DEL ANÁLISIS EXTENDIDO (DATOS REALES)")
    for img in ['comparativa_modelos_tfg.png', 'comparativa_iv_tfg.png',
                'convergencia_mc_ic_tfg.png', 'oos_multihorizonte_tfg.png',
                'superficie_smiles_tfg.png', 'superficie_3d_tfg.png',
                'term_structure_atm_tfg.png']:
        print(f"  - {img}")


if __name__ == "__main__":
    run()
