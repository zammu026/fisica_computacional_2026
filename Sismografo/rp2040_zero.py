from time import sleep_ms, ticks_ms, ticks_diff, localtime
from math import sqrt
from machine import Pin
from neopixel import NeoPixel
from MPU6050 import MPU6050
import gc

# === Configuración LED RGB integrado (WS2812 en GPIO16) ===
PIN_RGB = 16
np = NeoPixel(Pin(PIN_RGB), 1)

LED_VERDE    = (0, 50, 0)
LED_AMARILLO = (50, 30, 0)
LED_ROJO     = (50, 0, 0)
LED_MAGENTA  = (50, 0, 50)
LED_APAGADO  = (0, 0, 0)

def set_led(color):
    """Actualiza el color del LED RGB."""
    np[0] = color
    np.write()

# === Configuración acelerómetro y parámetros del sistema ===
G = 9.80665  # gravedad [m/s^2]
mpu = MPU6050()

LECTURAS_HZ      = 100
PERIODO_MS       = int(1000 / LECTURAS_HZ)

UMBRAL_INICIO    = 0.03
UMBRAL_FIN       = 0.01
PRE_EVENTO       = 100
POST_EVENTO      = 2000
INACTIVIDAD_MS   = 500

BLOQUE_RAM       = 200
MAX_RAM          = 5000
LIMITE_SEGMENTO  = 10000

# === Funciones principales ===

def calibrar_acelerometro(duracion_ms=10000, intervalo_ms=10):
    """Calcula offsets promedio del sensor acelerómetro manteniéndolo inmóvil."""
    print("Calibrando... mantén el sensor inmóvil.")
    start = ticks_ms()
    sum_x = sum_y = sum_z = 0.0
    n = 0
    while ticks_diff(ticks_ms(), start) < duracion_ms:
        acc = mpu.read_accel_data()
        sum_x += acc["x"]
        sum_y += acc["y"]
        sum_z += acc["z"]
        n += 1
        sleep_ms(intervalo_ms)
    offsets = {"x": (sum_x/n)/G, "y": (sum_y/n)/G, "z": (sum_z/n)/G}
    print("Calibración completa (en g):", offsets)
    return offsets

def generar_nombre_archivo():
    """Genera nombre de archivo basado en fecha y hora actual."""
    y, m, d, h, mi, s, *_ = localtime()
    return f"evento_{y}-{m:02d}-{d:02d}_{h:02d}-{mi:02d}-{s:02d}.txt"

def guardar_bloque(nombre, bloque):
    """Guarda un bloque de datos en flash, liberando memoria inmediatamente."""
    if not bloque: 
        return
    try:
        with open(nombre, "a") as f:
            for linea in bloque:
                f.write(linea)
        print(f"[OK] Guardado {len(bloque)} muestras en {nombre}")
    except Exception as e:
        print("Error al escribir en flash:", e)
    gc.collect()

def leer_acelerometro(offsets):
    """Lee acelerómetro y devuelve valores normalizados y magnitud."""
    acc = mpu.read_accel_data()
    aX = (acc["x"]/G) - offsets["x"]
    aY = (acc["y"]/G) - offsets["y"]
    aZ = (acc["z"]/G) - offsets["z"]
    mag = sqrt(aX*aX + aY*aY + aZ*aZ)
    return aX, aY, aZ, mag

def procesar_evento(buffer_pre, buffer_ram, nombre, t_evento_inicio, t_abs, aX, aY, aZ):
    """Agrega la muestra actual al buffer RAM con tiempo relativo."""
    t_rel = t_abs - t_evento_inicio
    buffer_ram.append(f"{t_rel:.3f},{aX:.4f},{aY:.4f},{aZ:.4f}\n")

def manejar_segmentacion(buffer_ram, nombre, segmento):
    """Crea un nuevo segmento si el evento supera el límite de líneas."""
    guardar_bloque(nombre, buffer_ram)
    buffer_ram.clear()
    segmento += 1
    nuevo_nombre = nombre.replace(".txt", f"_{segmento:02d}.txt")
    print(f">> Nuevo segmento creado: {nuevo_nombre}")
    return nuevo_nombre, segmento

def control_buffer(buffer_ram, nombre, grabando):
    """Evita desbordamiento de RAM guardando datos de emergencia."""
    if len(buffer_ram) > MAX_RAM * 0.9:
        print("Advertencia: buffer RAM casi lleno, guardando de emergencia...")
        set_led(LED_MAGENTA)
        if nombre:
            guardar_bloque(nombre, list(buffer_ram))
        buffer_ram.clear()
        set_led(LED_ROJO if grabando else LED_VERDE)

# === Inicialización ===
offsets = calibrar_acelerometro()
set_led(LED_VERDE)
print("Esperando eventos...")

buffer_pre = []
buffer_ram = []
grabando = False
t_inicio_programa = ticks_ms()
nombre = None
t_evento_inicio = 0
ultimo_evento = ticks_ms()
conteo_evento = 0
total_eventos = 0
segmento = 0

# === Bucle principal ===
try:
    while True:
        t_abs = ticks_diff(ticks_ms(), t_inicio_programa)/1000.0
        aX, aY, aZ, mag = leer_acelerometro(offsets)

        # Buffer pre-evento
        buffer_pre.append((t_abs, aX, aY, aZ))
        if len(buffer_pre) > PRE_EVENTO:
            buffer_pre.pop(0)

        # --- Inicio de evento ---
        if not grabando and mag > UMBRAL_INICIO:
            y, m, d, h, mi, s, *_ = localtime()
            hora_evento = f"{h:02d}:{mi:02d}:{s:02d}"
            print(f">> Evento detectado ({hora_evento}), magnitud={mag:.3f} g")
            grabando = True
            t_evento_inicio = t_abs
            ultimo_evento = ticks_ms()
            conteo_evento = 0
            total_eventos += 1
            segmento = 0
            nombre = generar_nombre_archivo()

            set_led(LED_ROJO if total_eventos >= 3 else LED_AMARILLO)

            # Crear archivo con cabecera
            with open(nombre, "w") as f:
                f.write(f"# Evento iniciado a las {hora_evento}\n")
                f.write("# t_rel(s),aX(g),aY(g),aZ(g)\n")

            # Vaciar buffer pre-evento al buffer RAM
            for t_p, ax, ay, az in buffer_pre:
                procesar_evento(buffer_pre, buffer_ram, nombre, t_evento_inicio, t_p, ax, ay, az)
            buffer_pre.clear()

        # --- Grabación durante evento ---
        elif grabando:
            if mag > UMBRAL_FIN:
                ultimo_evento = ticks_ms()
            procesar_evento(buffer_pre, buffer_ram, nombre, t_evento_inicio, t_abs, aX, aY, aZ)
            conteo_evento += 1

            # Escritura diferida
            if len(buffer_ram) >= BLOQUE_RAM:
                guardar_bloque(nombre, buffer_ram)
                buffer_ram.clear()

            # Segmentación de evento largo
            if conteo_evento >= LIMITE_SEGMENTO:
                nombre, segmento = manejar_segmentacion(buffer_ram, nombre, segmento)
                conteo_evento = 0

            # Fin del evento
            if ticks_diff(ticks_ms(), ultimo_evento) > INACTIVIDAD_MS or conteo_evento >= POST_EVENTO:
                print(">> Evento finalizado.")
                grabando = False
                if buffer_ram:
                    guardar_bloque(nombre, buffer_ram)
                    buffer_ram.clear()
                    print(f">> Datos guardados en '{nombre}'")
                nombre = None
                if total_eventos < 3:
                    set_led(LED_VERDE)

        # Control de buffer global
        control_buffer(buffer_ram, nombre, grabando)
        sleep_ms(PERIODO_MS)

except KeyboardInterrupt:
    # Finalización segura
    set_led(LED_APAGADO)
    if buffer_ram and nombre:
        guardar_bloque(nombre, buffer_ram)
    print("\nMonitoreo detenido por el usuario.")
    print(f"Total de eventos registrados: {total_eventos}")