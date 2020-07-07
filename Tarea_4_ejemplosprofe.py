import numpy as np
from scipy import stats
from scipy import signal
from scipy import integrate
import matplotlib.pyplot as plt

# Número de bits
N = 100

# Variable aleatoria binaria equiprobable
X = stats.bernoulli(0.5)

# Generar bits para "transmitir"
bits = X.rvs(N)

'''

#   (20 %) Crear un esquema de modulación BPSK para los bits presentados. Esto implica asignar 
#   una forma de onda sinusoidal normalizada (amplitud unitaria)
#   para cada bit y luego una concatenación de todas estas formas de onda.
'''

# Frecuencia de operación
f = 1000 # Hz

# Duración del período de cada símbolo (onda)
T = 1/f # 1 ms

# Número de puntos de muestreo por período
p = 50

# Puntos de muestreo para cada período
tp = np.linspace(0, T, p)

# Creación de la forma de onda de la portadora
sinus = np.sin(2*np.pi * f * tp)

# Visualización de la forma de onda de la portadora
plt.plot(tp, sinus)
plt.xlabel('Tiempo / s')
plt.savefig('grafica1.png')

# Frecuencia de muestreo
fs = p/T # 50 kHz

# Creación de la línea temporal para toda la señal Tx
t = np.linspace(0, N*T, N*p)

# Inicializar el vector de la señal
senal = np.zeros(t.shape)

# Creación de la señal modulada OOK
for k, b in enumerate(bits):
  senal[k*p:(k+1)*p] = b * sinus

# Visualización de los primeros bits modulados

pb = 5
plt.figure()
plt.plot(senal[0:pb*p]) 
plt.savefig('Tx.png')


'''
2. Calcular la potencia promedio
'''

# Potencia instantánea
Pinst = senal**2

# Potencia promedio (W)
Ps = integrate.trapz(Pinst, t) / (N * T)

'''
3. Simular un canal ruidoso del tipo AWGN (ruido aditivo blanco gaussiano).
'''

# Relación señal-a-ruido deseada
SNR = 60

# Potencia del ruido para SNR y potencia de la señal dadas
Pn = Ps / (10**(SNR / 10))

# Desviación estándar del ruido
sigma = np.sqrt(Pn)

# Crear ruido (Pn = sigma^2)
ruido = np.random.normal(0, sigma, senal.shape)

# Simular "el canal": señal recibida
Rx = senal + ruido

# Visualización de los primeros bits recibidos
pb = 5
plt.figure()
plt.plot(Rx[0:pb*p])

plt.savefig('Rx.png')

'''
4. Demodular y decodificar la señal.
'''

# Pseudo-energía de la onda original
Es = np.sum(sinus**2)

# Inicialización del vector de bits recibidos
bitsRx = np.zeros(bits.shape)

# Decodificación de la señal por detección de energía
for k, b in enumerate(bits):
  # Producto interno de dos funciones
  Ep = np.sum(Rx[k*p:(k+1)*p] * sinus) 
  if Ep > Es/2:
    bitsRx[k] = 1
  else:
    bitsRx[k] = 0

err = np.sum(np.abs(bits - bitsRx))
BER = err/N
