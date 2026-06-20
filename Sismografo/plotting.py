import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === Nombre del archivo ===
archivo = "/content/evento_2025-11-13_11-17-50.txt"
# === Leer archivo de texto con encabezado ===
try:
    # Skip the first row and specify the delimiter
    datos = pd.read_csv(archivo, skiprows=1)
except Exception as e:
    print(f"Error al leer el archivo: {e}")
    raise

# === Comprobar columnas esperadas ===
expected = {"# t_rel(s)", "aX(g)", "aY(g)", "aZ(g)"} # Update expected column names
if not expected.issubset(datos.columns):
    print("Columnas encontradas:", datos.columns.tolist())
    raise SystemExit("El archivo debe contener las columnas: t_abs(s),aX(g),aY(g),aZ(g)")

# === Extraer columnas ===
t = datos["# t_rel(s)"].astype(float) # Update column name
aX = datos["aX(g)"].astype(float)   # Update column name
aY = datos["aY(g)"].astype(float)   # Update column name
aZ = datos["aZ(g)"].astype(float)   # Update column name

# === Calcular magnitud total de aceleración ===
mag = np.sqrt(aX**2 + aY**2 + aZ**2)

# === Mostrar primeras filas ===
print("\nPrimeras líneas del archivo:")
print(datos.head())

# === Gráfico eje X ===
plt.figure(figsize=(8, 4))
plt.plot(t, aX,color='green')
plt.title("Aceleración eje X")
plt.xlabel("Tiempo (s)")
plt.ylabel("aX (g)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfico eje Y ===
plt.figure(figsize=(8, 4))
plt.plot(t, aY,color='red')
plt.title("Aceleración eje Y")
plt.xlabel("Tiempo (s)")
plt.ylabel("aY (g)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfico eje Z ===
plt.figure(figsize=(8,4))
plt.plot(t, aZ,color='blue')
plt.title("Aceleración eje Z")
plt.xlabel("Tiempo (s)")
plt.ylabel("aZ (g)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfico magnitud total ===
plt.figure(figsize=(8, 4))
plt.plot(t, mag,color='black')
plt.title("Magnitud total de aceleración |a|")
plt.xlabel("Tiempo (s)")
plt.ylabel("|a| (g)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === FFT de la magnitud |a| ===

from scipy.fft import rfft, rfftfreq

# --- Calcular frecuencia de muestreo fs ---
dt = np.mean(np.diff(t))      # intervalo de tiempo promedio
fs = 1.0 / dt                 # frecuencia de muestreo

print(f"\nFrecuencia de muestreo estimada: {fs:.2f} Hz")

# --- FFT real ---
N = len(mag)                  # número de muestras
fft_vals = rfft(mag)          # FFT
fft_mag = np.abs(fft_vals)    # magnitud
freqs = rfftfreq(N, dt)       # eje de frecuencias

# --- Gráfica de espectro ---
plt.figure(figsize=(8, 4))
plt.plot(freqs, fft_mag,color='blue')
plt.title("Espectro FFT de la magnitud |a|")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (u.a.)")
plt.ylim(0, 14)
plt.grid(True)
plt.xlim(1, fs/2)             # límite útil: Nyquist
plt.tight_layout()
plt.show()
