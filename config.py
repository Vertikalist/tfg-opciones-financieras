import numpy as np

# CONFIGURACIÓN GENERAL DEL MERCADO
TICKER_S_P500 = "^SPX"          # S&P 500 (Ideal por volumen y liquidez)
TICKER_EUROSTOXX = "^STOXX50E"  # Euro Stoxx 50 (Alternativa europea)
TICKER_IBEX35 = "^IBEX"         # IBEX 35 (Alternativa española)

# Activo por defecto y tasa de interés libre de riesgo estimada
TICKER_ACTIVO = TICKER_S_P500
R_PERIODO =  "1mo"  # Tomamos como tasa libre de riesgo el tipo de interés del Bono del Tesoro Estadounidense
INDEX_VENCIMIENTO = 2           # Índice de fallback si ningún vencimiento cumple los criterios
MIN_DIAS_VENCIMIENTO = 20       # Mínimo de días hasta vencimiento (evita expiries casi expirados)
MIN_OPCIONES_LIQUIDAS = 8       # Mínimo de opciones con bid > 0 en el rango ±20% del spot

# PARAMETRIZACIÓN DE SIMULACIONES (MONTE CARLO)
NUM_SIMULACIONES = 100000        # Número de trayectorias aleatorias a generar
NUM_PASOS_TIEMPO = 252          # Pasos discretos por año (días laborables)

# CONFIGURACIÓN DE LOS ALGORITMOS DE OPTIMIZACIÓN
# Límites (Bounds) para Merton: [sigma, lambd, mu_j, sigma_j]
BOUNDS_MERTON = [(0.01, 1.0), (0.0, 5.0), (-0.5, 0.5), (0.001, 0.5)]

# Límites (Bounds) para Heston: [kappa, theta, xi, rho, v0]
BOUNDS_HESTON = [(0.1, 10.0), (0.001, 1.0), (0.01, 2.0), (-0.95, 0.0), (0.001, 1.0)]

# Configuración del algoritmo de Evolución Diferencial (mantenido por compatibilidad)
DE_MAX_ITER = 50                # Máximo de generaciones del algoritmo genético
DE_POP_SIZE = 10                # Multiplicador del tamaño de la población
SEMILLA_ALEATORIA = 42          # Asegura la reproducibilidad de los resultados

# Configuración de L-BFGS-B con reinicios aleatorios
N_RESTARTS = 5                 # Número de puntos de inicio aleatorios para evitar mínimos locales

# Peso de la penalización matemática si se viola la Condición de Feller (2*kappa*theta > xi^2)
PENALIZACION_FELLER = 1000.0
