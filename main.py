import pandas as pd
from config import R_PERIODO, NUM_SIMULACIONES, NUM_PASOS_TIEMPO
from modelos import ModelosValoracion
from datos_mercado import obtener_datos_mercado, get_risk_free_rate
from calibration import calibrar_black_scholes, calibrar_merton, calibrar_heston
from utils import generar_graficos_comparativos, generar_graficos_comparativos_iv, graficar_convergencia_monte_carlo
from validacion_oos import ejecutar_validacion_oos
def run_pipeline():
    # 1. Obtención de datos reales de mercado
    tasa_r = get_risk_free_rate(R_PERIODO)
    S_mercado, T_mercado, df_opciones = obtener_datos_mercado()
    calc = ModelosValoracion(r=tasa_r)
    print("="*60)
    print("EJECUTANDO FLUJO MODULAR DEL TFG")
    print("="*60)
    print(f"Subyacente S: {S_mercado:.2f} | Vencimiento T: {T_mercado:.4f} años\n")
    # 3. Calibraciones (In-Sample)
    print("-> Calibrando Precio Black-Scholes...")
    res_bs = calibrar_black_scholes(S_mercado, T_mercado, df_opciones, calc)
    print("-> Calibrando Precio Merton...")
    res_merton = calibrar_merton(S_mercado, T_mercado, df_opciones, calc)
    print("-> Calibrando Precio Heston...")
    res_heston = calibrar_heston(S_mercado, T_mercado, df_opciones, calc)

    print("-> Calibrando Volatilidad Implícita Black-Scholes...")
    res_bs = calibrar_black_scholes(S_mercado, T_mercado, df_opciones, calc)
    print("-> Calibrando Volatilidad Implícita Merton...")
    res_merton = calibrar_merton(S_mercado, T_mercado, df_opciones, calc)
    print("-> Calibrando Volatilidad Implícita  Heston...")
    res_heston = calibrar_heston(S_mercado, T_mercado, df_opciones, calc)

    # 3. Resumen de Métricas
    print("\n" + "="*60)
    print("TABLA COMPARATIVA FINAL DE RENDIMIENTO (IN-SAMPLE)")
    print("="*60)
    tabla_resultados = pd.DataFrame({
    'Modelo': ['Black-Scholes', 'Merton', 'Heston'],
    'MSE Final': [res_bs.fun, res_merton.fun, res_heston.fun],
    'Parámetros': [1, 4, 5]
    })
    print(tabla_resultados.to_string(index=False))
    # 4. Validación Numérica con Monte Carlo
    print("\n" + "="*60)
    print("VALIDACIÓN NUMÉRICA: FOURIER VS MONTE CARLO (HESTON)")
    print("="*60)
    kap_h, the_h, xi_h, rho_h, v0_h = res_heston.x
    strike_test = df_opciones.iloc[len(df_opciones)//2]['K']
    precio_mkt_test = df_opciones.iloc[len(df_opciones)//2]['C_mkt']
    precio_fourier = calc.precio_heston(S_mercado, strike_test, T_mercado, kap_h, the_h, xi_h, rho_h, v0_h)
    print(f"Computando {NUM_SIMULACIONES} trayectorias...")
    precio_mc = calc.precio_heston_monte_carlo(S_mercado, strike_test, T_mercado, kap_h, the_h, xi_h, rho_h, v0_h,
    NUM_SIMULACIONES, NUM_PASOS_TIEMPO)
    print(f"\nResultados para Strike K = {strike_test}:")
    print(f" - Precio Mercado: {precio_mkt_test:.2f}")
    print(f" - Heston Fourier: {precio_fourier:.2f}")
    print(f" - Heston Monte Carlo: {precio_mc:.2f}")
    print(f" - Error numérico: {abs(precio_fourier - precio_mc):.4f}")
    print("="*60)
    # 5. Generar Visualizaciones
    print("\n-> Generando gráficos estéticos con Seaborn...")
    graficar_convergencia_monte_carlo(S_mercado, strike_test, T_mercado, calc, res_heston.x, NUM_PASOS_TIEMPO)

    generar_graficos_comparativos(df_opciones, S_mercado, T_mercado, calc, res_bs.x, res_merton.x, res_heston.x)

    generar_graficos_comparativos_iv(df_opciones, S_mercado, T_mercado, calc, res_bs.x, res_merton.x, res_heston.x)

    # 6. Validación estática OOS (siguiente expiry disponible)
    ejecutar_validacion_oos(S_mercado, T_mercado, calc, res_bs, res_merton, res_heston)

if __name__ == "__main__":
    run_pipeline()
