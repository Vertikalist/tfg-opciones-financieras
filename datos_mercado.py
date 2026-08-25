import numpy as np
import yfinance as yf
import pandas as pd
from config import TICKER_ACTIVO, INDEX_VENCIMIENTO, MIN_DIAS_VENCIMIENTO, MIN_OPCIONES_LIQUIDAS


def obtener_datos_mercado(ticker_symbol=TICKER_ACTIVO, index_vencimiento=INDEX_VENCIMIENTO):
    ticker = yf.Ticker(ticker_symbol)
    S = ticker.history(period="1d")['Close'].iloc[-1]

    fecha_hoy = pd.Timestamp.today()
    vencimientos = ticker.options

    # Buscar el primer vencimiento con al menos MIN_DIAS días y MIN_OPCIONES opciones limpias.
    # Esto evita coger vencimientos casi expirados donde solo 1-2 strikes tienen bid > 0.
    MIN_DIAS = MIN_DIAS_VENCIMIENTO
    MIN_OPCIONES = MIN_OPCIONES_LIQUIDAS

    df_calib = pd.DataFrame()
    fecha_vencimiento = None
    T = None

    candidatos = [v for v in vencimientos
                  if (pd.to_datetime(v) - fecha_hoy).days >= MIN_DIAS]

    # Si no hay candidatos con el mínimo de días, usar INDEX_VENCIMIENTO como fallback
    if not candidatos:
        candidatos = [vencimientos[index_vencimiento]]

    for fecha in candidatos:
        T_candidato = (pd.to_datetime(fecha) - fecha_hoy).days / 365.0
        calls = ticker.option_chain(fecha).calls
        calls['Mid_Price'] = (calls['bid'] + calls['ask']) / 2

        datos = calls[(calls['bid'] > 0) & (calls['openInterest'] > 0) &
                      (calls['strike'] > 0.8 * S) & (calls['strike'] < 1.2 * S)].copy()

        if len(datos) >= MIN_OPCIONES:
            df_calib = datos[['strike', 'Mid_Price']].rename(columns={'strike': 'K', 'Mid_Price': 'C_mkt'})
            fecha_vencimiento = fecha
            T = T_candidato
            break

    if df_calib.empty:
        raise ValueError(
            f"No se encontró ningún vencimiento con >= {MIN_OPCIONES} opciones líquidas "
            f"a más de {MIN_DIAS} días. Vencimientos disponibles: {list(vencimientos)}"
        )

    # Ajuste por dividendos continuos: S_adj = S * exp(-q * T)
    # Permite usar modelos sin dividendos explícitos con el precio forward correcto.
    q = get_dividend_yield(ticker_symbol)
    S_adj = S * np.exp(-q * T)

    print(f"   Vencimiento seleccionado: {fecha_vencimiento} | T = {T:.4f} años | {len(df_calib)} opciones")
    print(f"   Dividend yield: {q*100:.2f}% | S = {S:.2f} | S_adj = {S_adj:.2f}")
    return S_adj, T, df_calib


def get_dividend_yield(ticker_symbol=TICKER_ACTIVO):
    """
    Obtiene la rentabilidad por dividendo (q) anualizada del activo.
    Para índices como el S&P 500, los dividendos se aproximan como continuos.
    Devuelve 0.0 si no está disponible.
    """
    try:
        info = yf.Ticker(ticker_symbol).info
        q = info.get('dividendYield', 0.0) or 0.0
        return float(q)
    except Exception:
        return 0.0


def get_implied_volatility_surface(ticker):

    # 1. Fetch the ticker
    ticker = yf.Ticker("SPY")
    underlying_price = ticker.fast_info['lastPrice']

    # 2. Get all available expiration dates
    expirations = ticker.options

    all_options = []

    # 3. Loop through expiries to build the surface dataset
    for expiry in expirations:
        # Calculate Time to Maturity (T) in years
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        days_to_maturity = (expiry_date - datetime.now()).days
        T = days_to_maturity / 365.0

        # Filter out options expiring too soon (less than 7 days) to avoid noise
        if T < (7 / 365.0):
            continue

        # Fetch call options chain
        opt_chain = ticker.option_chain(expiry)
        calls = opt_chain.calls

        # Clean and filter the data
        for _, row in calls.iterrows():
            # Keep liquid options with open interest and reasonable volume
            if row['openInterest'] > 10 and row['impliedVolatility'] > 0.01:
                all_options.append({
                    'Strike': row['strike'],
                    'Maturity_T': T,
                    'Market_Price': (row['bid'] + row['ask']) / 2,  # Mid-price
                    'Implied_Vol': row['impliedVolatility'],
                    'Underlying_Price': underlying_price
                })

    # 4. Convert to a structured DataFrame for your calibration engine
    surface_df = pd.DataFrame(all_options)
    print(surface_df.head())
    return surface_df

#RISK FREE RATE#
#RISK FREE RATE#
def get_risk_free_rate(r_period):
    # Fetch the 10-Year Treasury Yield (^TNX) and convert to decimal annual rate
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period=r_period)
    data['Annual_Rate'] = data['Close'] / 100.0
    return data['Annual_Rate'].iloc[-1]


get_risk_free_rate("1mo")
