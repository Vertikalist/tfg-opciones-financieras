import numpy as np
import pandas as pd
import yfinance as yf

from config import TICKER_ACTIVO
from datos_mercado_old import get_dividend_yield


def _limpiar_calls(calls, S, rango=(0.8, 1.2)):
    """Limpia una cadena de calls: filtra liquidez/strikes y calcula el mid-price."""
    calls = calls.copy()
    calls['Mid_Price'] = (calls['bid'] + calls['ask']) / 2.0
    lo, hi = rango
    datos = calls[(calls['bid'] > 0) & (calls['openInterest'] > 0) &
                  (calls['strike'] > lo * S) & (calls['strike'] < hi * S)].copy()
    return datos[['strike', 'Mid_Price']].rename(columns={'strike': 'K', 'Mid_Price': 'C_mkt'})


def obtener_cadenas(ticker_symbol=TICKER_ACTIVO, min_dias=20, min_opciones=8,
                     max_vencimientos=6, rango=(0.8, 1.2)):
    """
    Descarga varias cadenas de calls (una por vencimiento) desde Yahoo Finance.

    Devuelve:
      S        : precio spot del subyacente
      q        : dividend yield anual
      cadenas: lista de tuplas (fecha_str, T_años, df) ordenada por vencimiento

    Parámetros:
      min_dias        : descarta vencimientos a menos de estos días (evita ruido)
      min_opciones    : nº mínimo de opciones líquidas para aceptar el vencimiento
      max_vencimientos: nº máximo de vencimientos a devolver
    """
    ticker = yf.Ticker(ticker_symbol)
    S = float(ticker.history(period="1d")['Close'].iloc[-1])
    q = get_dividend_yield(ticker_symbol)
    hoy = pd.Timestamp.today()

    cadenas = []
    for fecha in ticker.options:
        dias = (pd.to_datetime(fecha) - hoy).days
        if dias < min_dias:
            continue
        T = dias / 365.0
        try:
            df = _limpiar_calls(ticker.option_chain(fecha).calls, S, rango)
        except Exception:
            continue
        if len(df) >= min_opciones:
            cadenas.append((fecha, T, df.reset_index(drop=True)))
        if len(cadenas) >= max_vencimientos:
            break

    if not cadenas:
        raise ValueError(
            f"No se encontraron vencimientos con >= {min_opciones} opciones líquidas "
            f"a más de {min_dias} días para {ticker_symbol}."
        )

    print(f"  [YF] {ticker_symbol}: S = {S:.2f} | q = {q*100:.2f}% | "
          f"{len(cadenas)} vencimientos descargados")
    for fecha, T, df in cadenas:
        print(f"        - {fecha} | T = {T:.4f} años | {len(df)} opciones")
    return S, q, cadenas


def separar_insample_oos(cadenas):
    """
    A partir de la lista de cadenas, separa el primer vencimiento (in-sample) del
    resto (horizontes OOS).

    Devuelve: (T1, df1 insample), [(T oos, df oos), ...]
    """
    (_, T1, df1) = cadenas[0]
    horizontes = [(T, df) for (_, T, df) in cadenas[1:]]
    return (T1, df1), horizontes


def construir_superficie(cadenas):
    """Devuelve la superficie como lista de tuplas (T, df) para todos los vencimientos."""
    return [(T, df) for (_, T, df) in cadenas]
