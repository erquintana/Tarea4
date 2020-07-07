"""
TAREA #4
Esteban Leonardo Rodríguez Quinatna
B66076 
"""


import numpy as np
from scipy import signal
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd

#Extraccion de datos del bits10k.csv
datos= pd.read_csv('bits10k.csv', header= None, prefix="datos")
bits=pd.DataFrame(datos)

#   1.Crear un esquema de modulación BPSK para los bits presentados. 
#   Esto implica asignar una forma de onda sinusoidal normalizada 
#   (amplitud unitaria) para cada bit y luego una concatenación de 
#   todas estas formas de onda.
#Numero de bits
N = len(bits) 
 
#Frecuencia
f = 5000
#Duración del periodo de cada simbolo
T = 1/f

#Frecuencia de muestreo = frec de nyquist y puntos de muestra
#Numero de puntos de muestreo
p = 100
#Puntos de muestreo para cada periodo
tp = np.linspace(0, T, p)
# Creacion de la forma de onda de la portadora
sinus = np.sin(2*np.pi * f * tp)
# Visualizacion de la forma de onda de la portadora
plt.plot(tp,sinus)
plt.title('Forma de la onda portadora')
plt.xlabel('Tiempo / s')
plt.savefig("ondaportadora.png")
plt.show()

#Frecuencia de muestreo
fs = p/T
#Creacion de la linea temporal para toda la señal Tx
t = np.linspace(0, N*T, N*p)
#Inicializar el vector de la señal
senal = np.zeros(t.shape)
portadora = np.sin(2*np.pi*f*t)
#Creacion de la señal modulada
for k, b in enumerate(bits['datos0']):
    if b == 1:
        senal[k*p:(k+1)*p] = portadora[k*p:(k+1)*p]
    else:
        senal[k*p:(k+1)*p] = -portadora[k*p:(k+1)*p]
#Visualización de los primeros 5 bits modelados
pb=5
plt.plot(senal[0:pb*p])
plt.xlabel("Tiempo / s")
plt.ylabel("Amplitud")
plt.title("Visualización de los primeros 5 bits modelados")
plt.savefig("Tx.png")
plt.show()

#2.Calcular la potencia promedio de la señal modulada generada.

Pinst = senal**2
Pprom = integrate.trapz(Pinst, t)/(N*T)
print("La potencia promedio de señal modulada generada es: " + str(Pprom))

#   3.Simular un canal ruidoso del tipo AWGN (ruido aditivo blanco gaussiano) 
#   con una relación señal a ruido (SNR) desde -2 hasta 3 dB.
#   Para la relación de -2 SNR
SNR = -2 
Pruido = Pprom/(10**(SNR/10)) 
#Crear ruido (Pn=sigma^2)
sigma = np.sqrt(Pruido)
ruido = np.random.normal(0,sigma,senal.shape)

#Simular "el canal": señal recibida
Rx = senal + ruido

# Visualización del ruido
pb = 5
plt.title('Canal ruidoso con relación señal ruido SNR = -2')
plt.xlabel('Tiempo / s')
plt.plot(Rx[0:pb*p])
plt.savefig("SNR=-2.png")
plt.show()

#Para la relación de -1 SNR
SNR2 = -1
Pruido2 = Pprom/(10**(SNR2/10))  
#Crear ruido (Pn=sigma^2)
sigma2 = np.sqrt(Pruido2)
ruido2 = np.random.normal(0,sigma2,senal.shape)

#Simular "el canal": señal recibida
Rx2 = senal + ruido2

# Visualización del ruido
pb = 5
plt.title('Canal ruidoso con relación señal ruido SNR = -1')
plt.xlabel('Tiempo / s')
plt.plot(Rx2[0:pb*p])
plt.savefig("SNR=-1.png")
plt.show()

#Para la relación de 0 SNR

SNR3 = 0 
Pruido3 = Pprom/(10**(SNR3/10)) 
#Crear ruido (Pn=sigma^2)
sigma3 = np.sqrt(Pruido3)
ruido3 = np.random.normal(0,sigma3,senal.shape)

#Simular "el canal": señal recibida
Rx3 = senal + ruido3

# Visualización del ruido
pb = 5
plt.title('Canal ruidoso con relación señal ruido SNR = 0')
plt.xlabel('Tiempo / s')
plt.plot(Rx3[0:pb*p])
plt.savefig("SNR=0.png")
plt.show()

#Para la relación de 1 SNR

SNR4 = 1 #relación señal-ruido deseada
Pruido4 = Pprom/(10**(SNR4/10))  
#Crear ruido (Pn=sigma^2)
sigma4 = np.sqrt(Pruido4)
ruido4 = np.random.normal(0,sigma4,senal.shape)

#Simular "el canal": señal recibida
Rx4 = senal + ruido4

# Visualización del ruido
pb = 5
plt.title('Canal ruidoso con relación señal ruido SNR = 1')
plt.xlabel('Tiempo / s')
plt.plot(Rx4[0:pb*p])
plt.savefig("SNR=1.png")
plt.show()

# Para la relación de 2 SNR

SNR5 = 2 #relación señal-ruido deseada
Pruido5 = Pprom/(10**(SNR5/10))  
# Crear ruido (Pn=sigma^2)
sigma5 = np.sqrt(Pruido5)
ruido5 = np.random.normal(0,sigma5,senal.shape)

#Simular "el canal": señal recibida
Rx5 = senal + ruido5

# Visualización del ruido
pb = 5
plt.title('Canal ruidoso con relación señal ruido SNR = 2')
plt.xlabel('Tiempo / s')
plt.plot(Rx5[0:pb*p])
plt.savefig("SNR=2.png")
plt.show()

#Para la relación de 3 SNR señal-a-ruido deseada

SNR6 = 3 
#Potencia del ruido para SNR y potencia de la señal dadas
Pruido6 = Pprom/(10**(SNR6/10))  

#Desviaxión estándar del ruido
sigma6 = np.sqrt(Pruido6)

#Crear ruido (Pn=sigma^2)
ruido6 = np.random.normal(0,sigma6,senal.shape)

#Simular "el canal": señal recibida
Rx6 = senal + ruido6

# Visualización del ruido
pb = 5
plt.title('Canal ruidoso con relación señal ruido SNR = 6')
plt.xlabel('Tiempo / s')
plt.plot(Rx6[0:pb*p])
plt.savefig("SNR=3.png")
plt.show()

#4.Graficar la densidad espectral de potencia de la señal con el método 
#de Welch (SciPy), antes y después del canal ruidoso.
#Densidad espectral de potencia de la señal antes del canal ruidoso.
fwelch, Psd = signal.welch(senal, fs, nperseg=1024)
plt.title('Densidad espectral de potencia de la señal antes del canal ruidoso')
plt.semilogy(fwelch, Psd)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig("Antes")
plt.show()
#Densidad espectral de potencia de la señal después del canal ruidoso.
fwelch, Psd = signal.welch(Rx, fs, nperseg=1024)
plt.title('Densidad espectral de potencia de la señal después del canal ruidoso')
plt.semilogy(fwelch, Psd)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig("Después")
plt.show()


#   5.Demodular y decodificar la señal y hacer un conteo de la tasa de error 
#   de bits (BER, bit error rate) para cada nivel SNR.

# Pseudo-energía de la onda original
Es = np.sum(portadora**2)

# Inicialización del vector de bits recibidos
bitsRx = np.zeros(bits.shape)


#Para la relación de -2 SNR
for k, b in enumerate(bits):
    Ep = np.sum(Rx[k*p:(k+1)*p]*sinus)
    
    if Ep > Es/2:
        bitsRx[k] = 1
    else:
        bitsRx[k] = 0

error = np.sum(np.abs(bits - bitsRx)) 
BER = error/N
print('El error total es: ', error)
print('La tasa de error de bits para SNR= -2 es:',BER)

 
#Para la relación de -1 SNR
for k, b in enumerate(bits):
    Ep = np.sum(Rx2[k*p:(k+1)*p]*sinus)
    
    if Ep > Es/2:
        bitsRx[k] = 1
    else:
        bitsRx[k] = 0

error2 = np.sum(np.abs(bits - bitsRx)) 
BER2 = error/N
print('El error total es: ', error2)
print('La tasa de error de bits para SNR= -1 es: ',BER2) 

#Para la relación de 0 SNR
for k, b in enumerate(bits):
    Ep = np.sum(Rx3[k*p:(k+1)*p]*sinus)
    
    if Ep > Es/2:
        bitsRx[k] = 1
    else:
        bitsRx[k] = 0

error3 = np.sum(np.abs(bits - bitsRx)) 
BER3 = error/N
print('El error total es: ', error3)
print('La tasa de error de bits para SNR= 0 es: ',BER3) 


#Para la relación de 1 SNR
for k, b in enumerate(bits):
    Ep = np.sum(Rx4[k*p:(k+1)*p]*sinus)
    
    if Ep > Es/2:
        bitsRx[k] = 1
    else:
        bitsRx[k] = 0

error4 = np.sum(np.abs(bits - bitsRx)) 
BER4 = error/N
print('El error total es: ', error4)
print('La tasa de error de bits para SNR= 1 es: ',BER4) 

#Para la relación de 2 SNR
for k, b in enumerate(bits):
    Ep = np.sum(Rx5[k*p:(k+1)*p]*sinus)
    
    if Ep > Es/2:
        bitsRx[k] = 1
    else:
        bitsRx[k] = 0

error5 = np.sum(np.abs(bits - bitsRx)) 
BER5 = error/N
print('El error total es: ', error5)
print('La tasa de error de bits para SNR= 2 es: ',BER5) 


#Para la relación de 3 SNR
for k, b in enumerate(bits):
    Ep = np.sum(Rx6[k*p:(k+1)*p]*sinus)
    
    if Ep > Es/2:
        bitsRx[k] = 1
    else:
        bitsRx[k] = 0

error6 = np.sum(np.abs(bits - bitsRx)) 
BER6 = error/N
print('El error total es: ', error6)
print('La tasa de error de bits para SNR= 3 es: ',BER6) 


#   6. Graficar BER versus SNR
#   Vectores para graficar
VectorBER = [BER, BER2, BER3, BER4, BER5, BER6]
VectorSNR = [-2,-1,0,1,2,3]

plt.title('Bit error rate para cada SNR')
plt.ylabel('BER')
plt.xlabel('SNR')
plt.plot(VectorBER,VectorSNR)
plt.savefig("BER-SNR")
plt.show()
