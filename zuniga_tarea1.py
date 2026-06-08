# Comunicaciones Opticas: Tarea 1
# Alumno: Zúñiga, Guillermo Rubén Darío
# Profesor: Rainero, Franciso
# 
# Se barrió la performance de las siguientes modulaciones:
#   PAM-2   M=2
#   PAM-4   M=4
#   QPSK    M=4
#   8-PSK   M=8
#   QAM-16  M=16
# 
# OBSERVACION EN LOS RESULTADOS:
# Se observa en QPSK y QAM-16 que, para SNR chico, la BER medida es menor a la teorica. Esto se debe a que se calcula la BER
# de forma aproximada como SER/k, sin tenerse en cuenta los errores donde cambian 2 o mas bits. 

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

L = 100000  # Longitud de simulacion. Cantidad de simbolos transmitidos
BR = 32e9   # Baud Rate
M_v = [2,4,4,8,16]       # Ordenes de modulacion (PAM-2, PAM-4, QPSK, 8-PSK, QAM-16)
snr_db_v = np.linspace(0, 20, 21) # Array de SNR [0,20] [dB]

fs = BR     # Frecuencia de muestreo
T = 1/BR    # Período de simbolo
Ts = 1/fs   # Periodo de muestreo

k_v = np.array(np.log2(M_v), np.int32) # N bits en cada modulacion

# Generar bitstream
N_bits_v = k_v * L # Cantidad de bits enviados por modulacion
bits_v = np.random.randint(0, 2, N_bits_v[len(M_v) - 1])
# Agrupar bits cada k
bits_reshaped_m = [[0]]*len(M_v)

# Agrupar bits para luego formar los simbolos
for i in range(len(M_v)):
    bits_reshaped_m[i] = bits_v[:N_bits_v[i]].reshape((L,k_v[i]))

# Funciones de mapeo de simbolos
def bits_to_pam2(b0):
    if b0==0:
        return -1
    if b0==1:
        return 1
    
def bits_to_pam4(b0, b1):
    if(b1,b0)==(0,0):
        return -3
    if(b1,b0)==(0,1):
        return -1
    if(b1,b0)==(1,0):
        return 3
    if(b1,b0)==(1,1):
        return 1

def bits_to_qpsk(b0, b1):
    a_i = bits_to_pam2(b0)
    a_q = bits_to_pam2(b1)

    return a_i + 1j * a_q

def bits_to_8psk(b0, b1, b2):
    if(b2,b1,b0)==(0,0,0):
        return np.exp(1j * (-3/4*np.pi))
    if(b2,b1,b0)==(0,0,1):
        return np.exp(1j * (4/4*np.pi))
    if(b2,b1,b0)==(0,1,0):
        return np.exp(1j * (2/4*np.pi))
    if(b2,b1,b0)==(0,1,1):
        return np.exp(1j * (3/4*np.pi))
    if(b2,b1,b0)==(1,0,0):
        return np.exp(1j * (-2/4*np.pi))
    if(b2,b1,b0)==(1,0,1):
        return np.exp(1j * (-1/4*np.pi))
    if(b2,b1,b0)==(1,1,0):
        return np.exp(1j * (1/4*np.pi))
    if(b2,b1,b0)==(1,1,1):
        return np.exp(1j * (0/4*np.pi))
    
def bits_to_qam16(b0, b1, b2, b3):
    a_i = bits_to_pam4(b0,b1)
    a_q = bits_to_pam4(b2,b3)

    return a_i + 1j * a_q

# Arreglo de simbolos. Todos los simbolos en las 5 modulaciones estan en esta matriz
symb_m = [[bits_to_pam2(b)                    for b in bits_reshaped_m[0]], 
          [bits_to_pam4(b[0],b[1])            for b in bits_reshaped_m[1]],
          [bits_to_qpsk(b[0],b[1])            for b in bits_reshaped_m[2]],
          [bits_to_8psk(b[0],b[1],b[2])       for b in bits_reshaped_m[3]],
          [bits_to_qam16(b[0],b[1],b[2],b[3]) for b in bits_reshaped_m[4]]]

symb_m = np.array(symb_m) # Convertir a array de numpy

# Funcion para medir BER
def BER_check(snr_db):

    # SNR de dB a "veces"
    snr = 10**(snr_db/10)
    # Potencia de la señal
    Ps_v = [np.var(symb_m[i]) for i in range(len(M_v))]
    Ps_v = np.array(Ps_v) # Convertir a array de numpy
    # Potencia del ruido
    Pn_v = Ps_v/snr

    # Generar ruido. Se distribuye la mitad de la potencia en la parte real y la imaginaria
    noise_m = [np.sqrt(Pn_v[i]/2) * (np.random.randn(L) + 1j * np.random.randn(L)) for i in range(len(M_v))]

    noise_m = np.array(noise_m)

    # Inyectar ruido al stream
    symb_noisy_m = symb_m + noise_m

    # Funciones de Slicer
    def pam2_slicer(rx):
        if(rx.real >= 0):
            return 1
        else:
            return -1

    def pam4_slicer(rx):
        if rx.real >= 2:
            rx_hat = 3
        if rx.real < 2:
            rx_hat = 1
        if rx.real < 0:
            rx_hat = -1
        if rx.real < -2:
            rx_hat = -3

        return rx_hat

    def qpsk_slicer(rx):
        i_hat = pam2_slicer(rx.real)
        q_hat = pam2_slicer(rx.imag)

        return i_hat + 1j*q_hat

    def f8psk_slicer(rx):
        if np.angle(rx) >= -8/8*np.pi:
            rx_hat = np.exp(1j * (4/4*np.pi))
        if np.angle(rx) > -7/8*np.pi:
            rx_hat = np.exp(1j * (-3/4*np.pi))
        if np.angle(rx) > -5/8*np.pi:
            rx_hat = np.exp(1j * (-2/4*np.pi))
        if np.angle(rx) > -3/8*np.pi:
            rx_hat = np.exp(1j * (-1/4*np.pi))
        if np.angle(rx) > -1/8*np.pi:
            rx_hat = np.exp(1j * (0/4*np.pi))
        if np.angle(rx) > 1/8*np.pi:
            rx_hat = np.exp(1j * (1/4*np.pi))
        if np.angle(rx) > 3/8*np.pi:
            rx_hat = np.exp(1j * (2/4*np.pi))
        if np.angle(rx) > 5/8*np.pi:
            rx_hat = np.exp(1j * (3/4*np.pi))
        if np.angle(rx) > 7/8*np.pi:
            rx_hat = np.exp(1j * (4/4*np.pi))

        return rx_hat

    def qam16_slicer(rx):
        i_hat = pam4_slicer(rx.real)
        q_hat = pam4_slicer(rx.imag)

        return i_hat + 1j*q_hat

    # Arreglo de simbolos finales de las 5 modulaciones
    symb_hat_m = [[pam2_slicer(rx)  for rx in symb_noisy_m[0]],
                  [pam4_slicer(rx)  for rx in symb_noisy_m[1]],
                  [qpsk_slicer(rx)  for rx in symb_noisy_m[2]],
                  [f8psk_slicer(rx) for rx in symb_noisy_m[3]],
                  [qam16_slicer(rx) for rx in symb_noisy_m[4]]]
    
    symb_hat_m = np.array(symb_hat_m)

    # Contar errores
    errores_v = [np.sum(symb_hat_m[i] != symb_m[i]) for i in range(len(M_v))]
    errores_v = np.array(errores_v)

    # Calcular Symbol Error Rate SOLO para simulaciones con 100 errores o mas para tener buena estadistica. Sino se lo deja en 0
    SER_v = [0]*len(M_v)
    for i in range(len(M_v)):
        if errores_v[i] >= 100:
            SER_v[i] = errores_v[i] / L

    # Calcular Bit Error Rate
    BER_aprox_v = SER_v / k_v

    # Funcion Q
    def Q(x):
        return 0.5 * erfc(x / np.sqrt(2))

    ber_theo_v = [0]*len(M_v)

    # BER teórica PAM-2
    ber_theo_v[0] = Q(np.sqrt(2*snr))
    # BER teórica PAM-4
    ber_theo_v[1] = 3/4 * Q(np.sqrt(2/5 * snr))
    # BER teórica QPSK
    ber_theo_v[2] = Q(np.sqrt(snr))
    # BER teórica 8-PSK
    ber_theo_v[3] = 2/3 * Q(np.sqrt(2*snr) * np.sin(np.pi / 8))
    # BER teórica QAM-16
    ber_theo_v[4] = 3/4 * Q(np.sqrt(snr/5))

    return errores_v, SER_v, BER_aprox_v, ber_theo_v

err_m = [0]*len(snr_db_v)
SER_m = [0]*len(snr_db_v)
BER_aprox_m = [0]*len(snr_db_v)
BER_theo_m = [0]*len(snr_db_v)

for i in range(len(snr_db_v)):
    print('Simulando SNR = ', snr_db_v[i], '/', snr_db_v[len(snr_db_v) - 1])

    err_m[i], SER_m[i], BER_aprox_m[i], BER_theo_m[i] = BER_check(snr_db_v[i])


err_m = np.array(err_m)
SER_m = np.array(SER_m)
BER_aprox_m = np.array(BER_aprox_m)
BER_theo_m = np.array(BER_theo_m)

# Plottear resultados.
# Se observa en QPSK y QAM-16 que, para SNR chico, la BER medida es menor a la teorica. Esto se debe a que se calcula la BER de forma aproximada como SER/k, sin tenerse en cuenta los errores donde cambian 2 o mas bits. 

print('Plotteando...')

plt.figure()

plt.semilogy(snr_db_v, BER_aprox_m[:,0], color='blue'  , linewidth=1, linestyle='-', marker='o', label='PAM-2')
plt.semilogy(snr_db_v, BER_aprox_m[:,1], color='orange', linewidth=1, linestyle='-', marker='o', label='PAM-4')
plt.semilogy(snr_db_v, BER_aprox_m[:,2], color='green' , linewidth=1, linestyle='-', marker='o', label='QPSK')
plt.semilogy(snr_db_v, BER_aprox_m[:,3], color='red'   , linewidth=1, linestyle='-', marker='o', label='8-PSK')
plt.semilogy(snr_db_v, BER_aprox_m[:,4], color='purple', linewidth=1, linestyle='-', marker='o', label='QAM-16')

plt.semilogy(snr_db_v, BER_theo_m[:,0], color='blue'   , linewidth=0.8, linestyle='--', marker='', label='PAM-2 teorico')
plt.semilogy(snr_db_v, BER_theo_m[:,1], color='orange' , linewidth=0.8, linestyle='--', marker='', label='PAM-4 teorico')
plt.semilogy(snr_db_v, BER_theo_m[:,2], color='green'  , linewidth=0.8, linestyle='--', marker='', label='QPSK teorico')
plt.semilogy(snr_db_v, BER_theo_m[:,3], color='red'    , linewidth=0.8, linestyle='--', marker='', label='8-PSK teorico')
plt.semilogy(snr_db_v, BER_theo_m[:,4], color='purple' , linewidth=0.8, linestyle='--', marker='', label='QAM-16 teorico')

plt.title(f"BER con AWGN")
plt.xlabel("SNR [dB]")
plt.ylabel("Bit Error Rate")
plt.grid()
plt.ylim([100 / (L * 4), 1]) # limite minimo en: N_errores_minimos / (N_simbolos * k_maximo_(QAM-16))
plt.legend()
plt.show()
