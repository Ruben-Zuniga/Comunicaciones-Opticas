import matplotlib.pyplot as plt
import numpy as np
# import scipy as sp
from scipy import signal as sig
from scipy.fft import fft, fftfreq, fftshift

T = 2
delta_T = 0.001
f1 = 1
f2 = 3
fc = 100

N = T / delta_T
w1 = 2*np.pi * f1
w2 = 2*np.pi * f2
wc = 2*np.pi * fc

t_v = np.arange(0,N) * delta_T
x1_v = np.cos(t_v * w1)
x2_v = 2 * np.cos(t_v * w2)
c1_v = np.cos(t_v * wc)
c2_v = np.cos(t_v * wc + np.pi/2)

x1_mod_v = x1_v * c1_v
x2_mod_v = x2_v * c2_v

mod_v = x1_mod_v + x2_mod_v

delta_f = 2*np.pi / (delta_T * N)
f_v = (np.arange(0,N) - N/2) * delta_f
X1_v = fftshift(fft(x1_v))
X2_v = fftshift(fft(x2_v))
C1_v = fftshift(fft(c1_v))
C2_v = fftshift(fft(c2_v))
mod_f_v = fftshift(fft(mod_v))

# delay = 0.5  # delay in seconds
# delay_samples = int(delay / delta_T)
# delayed_x1_v = np.concatenate([np.zeros(delay_samples), x1_v[:-delay_samples]])

# fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, sharex=True)
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1, sharex=True)

ax1.plot(f_v, X1_v.real, f_v, X1_v.imag, linewidth=1, linestyle='-')
ax2.plot(f_v, C1_v.real, f_v, C1_v.imag, linewidth=1, linestyle='-')
ax3.plot(f_v, X2_v.real, f_v, X2_v.imag, linewidth=1, linestyle='-')
ax4.plot(f_v, C2_v.real, f_v, C2_v.imag, linewidth=1, linestyle='-')
ax5.plot(f_v, mod_f_v.real, f_v, mod_f_v.imag, linewidth=1, linestyle='-')

# ax1.set_title('Conmutacion Real')
ax1.set_ylabel('x1')
ax1.grid(True)

ax2.set_ylabel('c1')
ax2.grid(True)

ax3.set_ylabel('x2')
ax3.grid(True)

ax4.set_ylabel('c2')
ax4.grid(True)

ax5.set_ylabel('xmod')
ax5.set_xlabel('frecuencia')
ax5.grid(True)

plt.show()