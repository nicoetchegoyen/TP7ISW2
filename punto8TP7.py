import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Datos históricos (ejemplo)
t_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
e_data = np.array([8, 21, 25, 30, 25, 24, 17, 15, 11, 6])

# Definir el modelo PNR
def modelo_PNR(t, a, b, c):
    return a * t**b * np.exp(-c * t)

# # Ajustar el modelo a los datos históricos
params, _ = curve_fit(modelo_PNR, t_data, e_data)

# # Obtener los valores ajustados
a, b, c = params
# print(f'Parámetros ajustados: a={a}, b={b}, c={c}')

# # Crear un rango de tiempo para la curva ajustada
t_vals = np.linspace(1, 10, 100)

# # Graficar los datos históricos y el modelo ajustado
# plt.scatter(t_data, e_data, color='red', label='Datos históricos')
# plt.plot(t_vals, modelo_PNR(t_vals, a, b, c), label=f'Modelo ajustado a={a:.2f}, b={b:.2f}, c={c:.2f}')

# Esfuerzo total de 72 PM (proyecto nuevo)
PM_total = 72

# Distribuir el esfuerzo total usando el mismo modelo
distribucion_esfuerzo = modelo_PNR(t_vals, a, b, c)
distribucion_esfuerzo = PM_total * distribucion_esfuerzo / np.sum(distribucion_esfuerzo)

plt.plot(t_vals, distribucion_esfuerzo, label=f'Distribución de 72 PM', linestyle='dashed')

plt.xlabel('Tiempo (meses)')
plt.ylabel('Esfuerzo (PM)')
plt.title('Distribución del esfuerzo en personas-mes')
plt.legend()
plt.show()


# Multiplicar a por 4 y graficar el resultado
a_modificado = 4 * a
dist_e_x4 = modelo_PNR(t_vals, a_modificado, b, c)
dist_e_x4 = PM_total * dist_e_x4 / np.sum(dist_e_x4)

plt.plot(t_vals, dist_e_x4, label=f'Modificado a={a_modificado:.2f}', linestyle='dotted')
plt.legend()
plt.show()