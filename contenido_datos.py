import json
import os

RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_estudio.json")

# Nº de parámetros por modelo (fijo, independiente de los datos)
N_PARAMS = {"Black-Scholes": 1, "Merton": 4, "Heston": 5}

# ----------------------------------------------------------------------
# Valores de referencia (fallback) – coinciden con la ejecución documentada.
# ----------------------------------------------------------------------
DEFAULTS = {
    "meta": {
        "ticker": "(ejecución de referencia)",
        "fecha": "(fecha de referencia)",
        "S": 5000.0, "r": 0.04, "T1": 0.0822, "n_strikes": 23,
    },
    "insample_mse": {"Black-Scholes": 10.7506, "Merton": 1.2798, "Heston": 1.1107},
    "heston_1v": {"kappa": 1.310, "theta": 0.0167, "xi": 0.208, "rho": -0.950, "v0": 0.0459},
    "surface_heston": {"kappa": 1.120, "theta": 0.0395, "xi": 0.2975, "rho": -0.656, "v0": 0.0447},
    "fourier_mc": {"K_atm": 4845.0, "fourier": 226.81, "mc": 225.88,
                   "se": 0.70, "ic95": 1.37, "error": 0.94},
    "oos": [
        {"dias": 60, "modelo": "Black-Scholes", "rmse": 6.223, "iv_rmse_pct": 2.917},
        {"dias": 60, "modelo": "Merton", "rmse": 4.203, "iv_rmse_pct": 1.830},
        {"dias": 60, "modelo": "Heston", "rmse": 2.019, "iv_rmse_pct": 1.431},
        {"dias": 90, "modelo": "Black-Scholes", "rmse": 10.895, "iv_rmse_pct": 2.481},
        {"dias": 90, "modelo": "Merton", "rmse": 8.895, "iv_rmse_pct": 2.393},
        {"dias": 90, "modelo": "Heston", "rmse": 4.292, "iv_rmse_pct": 1.023},
        {"dias": 120, "modelo": "Black-Scholes", "rmse": 13.931, "iv_rmse_pct": 2.470},
        {"dias": 120, "modelo": "Merton", "rmse": 12.606, "iv_rmse_pct": 2.887},
        {"dias": 120, "modelo": "Heston", "rmse": 6.217, "iv_rmse_pct": 1.029},
    ],
    "surface_metrics": [
        {"dias": 60, "bs_rmse": 7.86, "merton_rmse": 7.86, "heston_rmse": 3.16, "heston_iv_rmse_pct": 2.16},
        {"dias": 90, "bs_rmse": 10.06, "merton_rmse": 10.06, "heston_rmse": 1.47, "heston_iv_rmse_pct": 0.62},
        {"dias": 120, "bs_rmse": 13.22, "merton_rmse": 13.22, "heston_rmse": 1.62, "heston_iv_rmse_pct": 0.31},
        {"dias": 180, "bs_rmse": 17.82, "merton_rmse": 17.82, "heston_rmse": 2.10, "heston_iv_rmse_pct": 0.29},
    ]
}


def guardar(resultados):
    with open(RUTA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"  [DATOS] Resultados guardados en {RUTA}")


def get():
    """Devuelve los resultados del JSON si existe; si no, los DEFAULTS."""
    if os.path.exists(RUTA):
        try:
            with open(RUTA, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULTS
    return DEFAULTS


def _heston_dict(x):
    kappa, theta, xi, rho, v0 = [float(v) for v in x]
    return {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0}


def construir_resultados(meta, res_bs, res_merton, res_heston, fourier_mc,
                          df_oos, params_surf, df_surf_met):
    """
    Arma el dict estándar de resultados a partir de los objetos del pipeline.

    - meta           : dict con ticker, fecha, S, r, T1, n_strikes
    - res_*          : resultados de scipy.optimize.minimize (tienen .x y .fun)
    - fourier_mc     : dict con K_atm, fourier, mc, se, ic95, error
    - df_oos         : DataFrame con columnas Horizonte_dias, Modelo, RMSE_precio, IV_RMSE_pct
    - params_surf    : dict devuelto por analisis_extendido.calibrar_superficie
    - df_surf_met    : DataFrame con Vencimiento_dias, Modelo, RMSE_precio, IV_RMSE_pct
    """

    oos = [{"dias": int(r.Horizonte_dias), "modelo": str(r.Modelo),
            "rmse": float(r.RMSE_precio), "iv_rmse_pct": float(r.IV_RMSE_pct)}
           for r in df_oos.itertuples()]

    clave = {"Black-Scholes": "bs", "Merton": "merton", "Heston": "heston"}
    surface_metrics = []
    for dias, g in df_surf_met.groupby("Vencimiento_dias"):
        fila = {"dias": int(dias)}
        for r in g.itertuples():
            k = clave[str(r.Modelo)]
            fila[f"{k}_rmse"] = float(r.RMSE_precio)
            if str(r.Modelo) == "Heston":
                fila["heston_iv_rmse_pct"] = float(r.IV_RMSE_pct)
        surface_metrics.append(fila)

    return {
        "meta": meta,
        "insample_mse": {
            "Black-Scholes": float(res_bs.fun),
            "Merton": float(res_merton.fun),
            "Heston": float(res_heston.fun),
        },
        "heston_1v": _heston_dict(res_heston.x),
        "surface_heston": _heston_dict(params_surf["heston"]),
        "fourier_mc": {k: float(v) for k, v in fourier_mc.items()},
        "oos": oos,
        "surface_metrics": surface_metrics,
    }
