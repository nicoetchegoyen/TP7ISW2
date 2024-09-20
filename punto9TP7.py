#*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
#* EffortModel
#* Programa para procesar modelos lineales mediante correlación por cuadrados mínimos
#* 
#* UADER - FCyT
#* Ingeniería de Software II
#*
#* Dr. Pedro E. Colla
#* copyright (c) 2023,2024
#*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
import numpy as np
import pandas as pd
import argparse
import statsmodels.api as sm
import sys
import os
import matplotlib.pyplot as plt

#*------------------------------------------------------------------------------------------------
#* Almacena dataset histórico
#*------------------------------------------------------------------------------------------------
data = {
    'LOC': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    'Esfuerzo': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
}

#*------------------------------------------------------------------------------------------------
#* Inicialización del programa
#*------------------------------------------------------------------------------------------------
version="7.0"
linear=False
exponential=False
os.system('cls')

#*------------------------------------------------------------------------------------------------
#* Procesa argumentos
#*------------------------------------------------------------------------------------------------
# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-v", "--version", required=False, help="version", action="store_true")
ap.add_argument("-x", "--exponential", required=False, help="Exponential model", action="store_true")
ap.add_argument("-l", "--linear", required=False, help="Linear model", action="store_true")
args = vars(ap.parse_args())

if args['version'] == True:
    print("Program %s version %s" % (sys.argv[0], version))
    sys.exit(0)

if args['linear'] == True:
    print("Program %s version %s" % (sys.argv[0], version))
    print("Linear correlation model selected")
    linear=True

if args['exponential'] == True:
    print("Program %s version %s" % (sys.argv[0], version))
    print("Exponential correlation model selected")
    exponential=True

if linear==False and exponential==False:
    print("Program %s version %s" % (sys.argv[0], version))
    print("Debe indicar modelo lineal (-l) o exponencial (-x) o ambos")

#*-----------------------------------------------------------------------------------------------
#* Definir dataset y procesar correlación entre LOC (complejidad) y Esfuerzo (PM)
#*-----------------------------------------------------------------------------------------------
df = pd.DataFrame(data)
correlation = df['LOC'].corr(df['Esfuerzo'])

#*------------------------------------------------------------------------------------------------
#* Procesa modelo lineal, usa numpy polyfit()
#*------------------------------------------------------------------------------------------------
if linear==True:
    a, b = np.polyfit(df['LOC'], df['Esfuerzo'], 1)
    R = np.corrcoef(df['LOC'], df['Esfuerzo'])
    R2 = R * R
    r_value = R2[1][0]

    print("Modelo lineal E=%.6f + %.6f*LOC" % (b, a))
    print("El R-squared=%.4f (lineal)" % (r_value))

    lbl = ("modelo lineal (R-Sq=%.2f)" % (r_value))
    plt.plot(df['LOC'], a * df['LOC'] + b, label=lbl, color='red')

#*------------------------------------------------------------------------------------------------
#* Procesa modelo exponencial, utiliza OLS fit()
#*------------------------------------------------------------------------------------------------
if exponential==True:
    df['logEsfuerzo'] = np.log(df['Esfuerzo'])
    df['logLOC'] = np.log(df['LOC'])

    X = df['logLOC']
    Y = df['logEsfuerzo']
    X = sm.add_constant(X)  # Añadir una constante para el intercepto

    mx = sm.OLS(Y, X).fit()
    print(mx.summary())

    k = np.exp(mx.params['const'])
    b = mx.params['logLOC']

    print("Modelo exponencial E=%.6f*(LOC^%.6f)" % (k, b))
    print("El R-squared=%.4f (exponencial)" % (mx.rsquared))

    lbl = ("modelo exponencial (R-Sq=%.2f)" % (mx.rsquared))
    plt.plot(df['LOC'], k * (df['LOC'] ** b), label=lbl, color='green')

#*------------------------------------------------------------------------------------------------
#* Hace plot del dataset histórico
#*------------------------------------------------------------------------------------------------
plt.scatter(df['LOC'], df['Esfuerzo'], label='Datos históricos')
plt.xlabel('Complejidad [LOC]')
plt.ylabel('Esfuerzo (persona-mes)')
plt.legend()
plt.show()

# Definir el tamaño-complejidad del nuevo proyecto
LOC_nuevem = 9100

# Estimar esfuerzo con el modelo seleccionado
if linear == True:
    e_nuevem = a * LOC_nuevem + b
    print(f"Esfuerzo estimado para LOC={LOC_nuevem} con modelo lineal: {e_nuevem:.2f} PM")
    plt.plot(LOC_nuevem, e_nuevem, 'ro', label=f'Estimación LOC={LOC_nuevem}')

if exponential == True:
    e_nuevem = k * (LOC_nuevem ** b)
    print(f"Esfuerzo estimado para LOC={LOC_nuevem} con modelo exponencial: {e_nuevem:.2f} PM")
    plt.plot(LOC_nuevem, e_nuevem, 'ro', label=f'Estimación LOC={LOC_nuevem}')

# Graficar el resultado comparando con los datos históricos y el modelo
plt.scatter(df['LOC'], df['Esfuerzo'], label='Datos históricos')
plt.xlabel('Complejidad [LOC]')
plt.ylabel('Esfuerzo (persona-mes)')
plt.legend()
plt.show()

# Definir el tamaño-complejidad del proyecto pequeño
LOC_dosc = 200

# Estimar esfuerzo con el modelo seleccionado
if linear == True:
    e_dosc = a * LOC_dosc + b
    print(f"Esfuerzo estimado para LOC={LOC_dosc} con modelo lineal: {e_dosc:.2f} PM")
    plt.plot(LOC_dosc, e_dosc, 'bo', label=f'Estimación LOC={LOC_dosc}')

if exponential == True:
    e_dosc = k * (LOC_dosc ** b)
    print(f"Esfuerzo estimado para LOC={LOC_dosc} con modelo exponencial: {e_dosc:.2f} PM")
    plt.plot(LOC_dosc, e_dosc, 'bo', label=f'Estimación LOC={LOC_dosc}')

# Graficar el resultado comparando con los datos históricos y el modelo
plt.scatter(df['LOC'], df['Esfuerzo'], label='Datos históricos')
plt.xlabel('Complejidad [LOC]')
plt.ylabel('Esfuerzo (persona-mes)')
plt.legend()
plt.show()

