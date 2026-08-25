import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from calibration import implied_vol_bs

def graficar_convergencia_monte_carlo(S, K, T, calc_model, params_heston, num_pasos):
    kappa, theta, xi, rho, v0 = params_heston
    precio_fourier = calc_model.precio_heston(S, K, T, kappa, theta, xi, rho, v0)
    rangos_sim = np.linspace(1000, 50000, 15, dtype=int)
    precios_mc = []
    print("-> Calculando nodos de convergencia para el gráfico...")
    for sim in rangos_sim:
        p_mc = calc_model.precio_heston_monte_carlo(S, K, T, kappa, theta, xi, rho, v0, sim, num_pasos)
        precios_mc.append(p_mc)
    df_conv = pd.DataFrame({'Simulaciones': rangos_sim,'Precio_MC': precios_mc,'Precio_Fourier': [precio_fourier] * len(rangos_sim)})
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 5.5))
    sns.lineplot(data=df_conv, x='Simulaciones', y='Precio_MC',
    color='#1f77b4', linewidth=2, marker='o', markersize=6, label='Precio Monte Carlo')
    plt.axhline(y=precio_fourier, color='r', linestyle='--', linewidth=2, label='Precio Exacto (Fourier)')
    plt.fill_between(df_conv['Simulaciones'], df_conv['Precio_MC'], precio_fourier,
    color='#1f77b4', alpha=0.15, label='Error de Muestreo')
    plt.title(f'Convergencia de Monte Carlo (Heston) para Strike K = {K}', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Número de Trayectorias Simuladas (Muestra)', fontsize=11)
    plt.ylabel('Precio de la Opción Call', fontsize=11)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='best')
    plt.tight_layout()
    plt.savefig('convergencia_monte_carlo_tfg.png', dpi=300)
    plt.show()
    print(" [Utils] Gráfico de convergencia guardado como 'convergencia_monte_carlo_tfg.png'\n")


def generar_graficos_comparativos(
    df_opciones,
    S_mercado,
    T_mercado,
    calc_model,
    params_bs,
    params_merton,
    params_heston
):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # =========================
    # Cálculo de precios teóricos
    # =========================

    df_graficos = df_opciones.copy()

    params_bs = float(np.asarray(params_bs).squeeze())

    df_graficos['Black-Scholes'] = df_graficos['K'].apply(
        lambda k: calc_model.precio_black_scholes(
            S_mercado, k, T_mercado, params_bs
        )
    )

    df_graficos['Merton'] = df_graficos['K'].apply(
        lambda k: calc_model.precio_merton(
            S_mercado, k, T_mercado, *params_merton
        )
    )

    df_graficos['Heston'] = df_graficos['K'].apply(
        lambda k: calc_model.precio_heston(
            S_mercado, k, T_mercado, *params_heston
        )
    )

    print(df_graficos.dtypes)

    print(df_graficos.head())

    print(df_graficos.iloc[0])

    # =========================
    # Configuración gráfica
    # =========================

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6),
        constrained_layout=True
    )

    # ==================================================
    # GRÁFICO 1: Mercado vs Modelos
    # ==================================================

    df_long = df_graficos.melt(
        id_vars=['K', 'C_mkt'],
        value_vars=['Black-Scholes', 'Merton', 'Heston'],
        var_name='Modelo',
        value_name='Precio_Teorico'
    )

    sns.scatterplot(
        ax=axes[0],
        data=df_graficos,
        x='K',
        y='C_mkt',
        color='black',
        marker='X',
        s=100,
        label='Mercado',
        zorder=5
    )

    sns.lineplot(
        ax=axes[0],
        data=df_long,
        x='K',
        y='Precio_Teorico',
        hue='Modelo',
        style='Modelo',
        palette='Set1',
        linewidth=2.5
    )

    axes[0].axvline(
        x=S_mercado,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        alpha=0.8
    )

    ymax = axes[0].get_ylim()[1]

    axes[0].annotate(
        'ATM',
        xy=(S_mercado, ymax * 0.92),
        ha='center',
        color='gray',
        fontsize=10
    )

    axes[0].set_title(
        'Ajuste de Modelos frente a Mercado',
        fontweight='bold'
    )

    axes[0].set_xlabel('Strike (K)')
    axes[0].set_ylabel('Precio Call')

    # ==================================================
    # GRÁFICO 2: Residuos
    # ==================================================

    df_errores = pd.DataFrame({
        'K': df_graficos['K']
    })

    df_errores['Black-Scholes'] = (
        df_graficos['C_mkt'] -
        df_graficos['Black-Scholes']
    )

    df_errores['Merton'] = (
        df_graficos['C_mkt'] -
        df_graficos['Merton']
    )

    df_errores['Heston'] = (
        df_graficos['C_mkt'] -
        df_graficos['Heston']
    )

    df_err_long = df_errores.melt(
        id_vars=['K'],
        value_vars=[
            'Black-Scholes',
            'Merton',
            'Heston'
        ],
        var_name='Modelo',
        value_name='Error'
    )

    sns.lineplot(
        ax=axes[1],
        data=df_err_long,
        x='K',
        y='Error',
        hue='Modelo',
        style='Modelo',
        marker='o',
        markersize=8,
        palette='Set1',
        linewidth=2.5
    )

    axes[1].axhline(
        y=0,
        color='black',
        linestyle='--',
        linewidth=1
    )

    axes[1].grid(
        True,
        linestyle=':',
        alpha=0.6
    )

    axes[1].set_title(
        'Análisis de Residuos por Strike',
        fontweight='bold'
    )

    axes[1].set_xlabel('Strike (K)')
    axes[1].set_ylabel('Error (Mercado - Modelo)')

    # ==================================================
    # Leyendas
    # ==================================================

    axes[0].legend(title='')
    axes[1].legend(title='')

    # ==================================================
    # Guardado y visualización
    # ==================================================

    plt.savefig(
        'comparativa_modelos_tfg.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

def generar_graficos_comparativos_iv(
    df_opciones,
    S_mercado,
    T_mercado,
    calc_model,
    params_bs,
    params_merton,
    params_heston
):
    """
    Genera dos gráficos en el mismo estilo que generar_graficos_comparativos,
    pero expresados en términos de Volatilidad Implícita (IV):
      - Panel izquierdo : Sonrisa de volatilidad (IV de mercado vs. modelos)
      - Panel derecho   : Residuos de IV por strike (IV_mkt - IV_modelo)
    Las filas cuya inversión BS no converge se descartan silenciosamente.
    """

    params_bs = float(np.asarray(params_bs).squeeze())
    r = calc_model.r

    df_iv = df_opciones[['K', 'C_mkt']].copy()

    # --- Volatilidades implícitas de mercado ---
    df_iv['IV_mkt'] = df_iv.apply(
        lambda row: implied_vol_bs(row['C_mkt'], S_mercado, row['K'], T_mercado, r),
        axis=1
    )

    # --- Volatilidades implícitas de cada modelo ---
    def _iv_modelo(precio_fn, K):
        precio = precio_fn(K)
        return implied_vol_bs(precio, S_mercado, K, T_mercado, r)

    df_iv['Black-Scholes'] = df_iv['K'].apply(
        lambda k: _iv_modelo(
            lambda k_: calc_model.precio_black_scholes(S_mercado, k_, T_mercado, params_bs), k
        )
    )
    df_iv['Merton'] = df_iv['K'].apply(
        lambda k: _iv_modelo(
            lambda k_: calc_model.precio_merton(S_mercado, k_, T_mercado, *params_merton), k
        )
    )
    df_iv['Heston'] = df_iv['K'].apply(
        lambda k: _iv_modelo(
            lambda k_: calc_model.precio_heston(S_mercado, k_, T_mercado, *params_heston), k
        )
    )

    # Descartar filas con NaN en IV de mercado y reindexar
    df_iv = df_iv.dropna(subset=['IV_mkt']).reset_index(drop=True)

    # =========================
    # Configuración gráfica
    # =========================

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6),
        constrained_layout=True
    )

    # ==================================================
    # GRÁFICO 1: Sonrisa de volatilidad
    # ==================================================

    df_long = df_iv.melt(
        id_vars=['K', 'IV_mkt'],
        value_vars=['Black-Scholes', 'Merton', 'Heston'],
        var_name='Modelo',
        value_name='IV_Modelo'
    ).dropna(subset=['IV_Modelo'])

    sns.scatterplot(
        ax=axes[0],
        data=df_iv,
        x='K',
        y='IV_mkt',
        color='black',
        marker='X',
        s=100,
        label='Mercado',
        zorder=5
    )

    sns.lineplot(
        ax=axes[0],
        data=df_long,
        x='K',
        y='IV_Modelo',
        hue='Modelo',
        style='Modelo',
        palette='Set1',
        linewidth=2.5
    )

    axes[0].axvline(
        x=S_mercado,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        alpha=0.8
    )

    ymax = axes[0].get_ylim()[1]

    axes[0].annotate(
        'ATM',
        xy=(S_mercado, ymax * 0.92),
        ha='center',
        color='gray',
        fontsize=10
    )

    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y * 100:.1f}%')
    )

    axes[0].set_title(
        'Sonrisa de Volatilidad: Modelos vs. Mercado',
        fontweight='bold'
    )

    axes[0].set_xlabel('Strike (K)')
    axes[0].set_ylabel('Volatilidad Implícita (IV)')

    # ==================================================
    # GRÁFICO 2: Residuos de IV
    # ==================================================

    df_errores_iv = pd.DataFrame({'K': df_iv['K']})

    for modelo in ['Black-Scholes', 'Merton', 'Heston']:
        df_errores_iv[modelo] = df_iv['IV_mkt'] - df_iv[modelo]

    df_err_long = df_errores_iv.melt(
        id_vars=['K'],
        value_vars=['Black-Scholes', 'Merton', 'Heston'],
        var_name='Modelo',
        value_name='Error_IV'
    ).dropna(subset=['Error_IV'])

    sns.lineplot(
        ax=axes[1],
        data=df_err_long,
        x='K',
        y='Error_IV',
        hue='Modelo',
        style='Modelo',
        marker='o',
        markersize=8,
        palette='Set1',
        linewidth=2.5
    )

    axes[1].axhline(
        y=0,
        color='black',
        linestyle='--',
        linewidth=1
    )

    axes[1].grid(
        True,
        linestyle=':',
        alpha=0.6
    )

    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y * 100:.2f}%')
    )

    axes[1].set_title(
        'Residuos de Volatilidad Implícita por Strike',
        fontweight='bold'
    )

    axes[1].set_xlabel('Strike (K)')
    axes[1].set_ylabel('Error IV (Mercado - Modelo)')

    # ==================================================
    # Leyendas
    # ==================================================

    axes[0].legend(title='')
    axes[1].legend(title='')

    # ==================================================
    # Guardado y visualización
    # ==================================================

    plt.savefig(
        'comparativa_iv_tfg.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    print(" [Utils] Gráfico de volatilidad implícita guardado como 'comparativa_iv_tfg.png'\n")
