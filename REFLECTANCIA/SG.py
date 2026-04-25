"""
Graficador Multi-Plot con suavizado Savitzky-Golay
Requiere: pip install matplotlib scipy numpy

Uso: python grafico_savgol.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
from scipy.signal import savgol_filter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os


# =============================================================================
# PALETA DE COLORES — edita aquí para cambiar el tema visual
# =============================================================================
BG      = "#0f1117"   # fondo general
PANEL   = "#1a1d27"   # fondo del panel lateral
PANEL2  = "#141720"   # fondo de cada fila de archivo
ACCENT  = "#4f9cf9"   # azul principal (botones, títulos)
ACCENT2 = "#f97b4f"   # naranja (color por defecto del suavizado)
TEXT    = "#e8eaf0"   # texto normal
SUBTEXT = "#7b8099"   # texto secundario / etiquetas
BORDER  = "#2a2d3e"   # bordes y separadores
SUCCESS = "#4fcf80"   # verde (mensaje de éxito)
DANGER  = "#f94f4f"   # rojo (eliminar)

# Colores automáticos que se asignan a cada serie cargada
# Índices pares  → línea cruda
# Índices impares → línea suavizada
COLORS = [
    "#4f9cf9", "#f97b4f",   # serie 0
    "#4fcf80", "#f9c84f",   # serie 1
    "#c44ff9", "#4ff9f0",   # serie 2
    "#f94f9c", "#9cf94f",   # serie 3
    "#f96060", "#4f6af9",   # serie 4
]


# =============================================================================
# LECTURA DE ARCHIVOS
# Lee un .txt / .csv / .dat con columnas numéricas separadas por tab, coma o espacio
# =============================================================================
def leer_archivo(ruta, col_x=0, col_y=1, saltear=0):
    """
    Parámetros:
        ruta    : ruta al archivo
        col_x   : índice de la columna X (0 = primera columna)
        col_y   : índice de la columna Y
        saltear : cuántas líneas iniciales ignorar (encabezados)

    Devuelve:
        (x, y) como arrays numpy, o lanza ValueError si algo falla
    """
    datos = []
    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        lineas = f.readlines()

    for i, linea in enumerate(lineas):
        if i < saltear:
            continue
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue

        # detectar separador automáticamente
        for sep in ["\t", ",", ";"]:
            if sep in linea:
                partes = linea.split(sep)
                break
        else:
            partes = linea.split()

        try:
            fila = [float(p.replace(",", ".")) for p in partes if p.strip()]
            if len(fila) > max(col_x, col_y):
                datos.append(fila)
        except ValueError:
            continue  # saltar líneas no numéricas

    if not datos:
        raise ValueError("No se encontraron datos numéricos.")

    arr = np.array(datos)
    return arr[:, col_x], arr[:, col_y]


# =============================================================================
# HELPERS DE INTERFAZ — funciones pequeñas para no repetir código de widgets
# =============================================================================
def hacer_entry(parent, variable, ancho=6):
    """Crea un campo de entrada pequeño y lo empaca a la izquierda."""
    e = tk.Entry(
        parent, textvariable=variable,
        bg=BORDER, fg=TEXT, insertbackground=ACCENT,
        font=("Consolas", 9), relief="flat", bd=0, width=ancho
    )
    e.pack(side="left", ipady=3, padx=(0, 3))
    return e

def etiqueta_seccion(parent, texto):
    """Etiqueta azul de sección dentro del panel."""
    tk.Label(
        parent, text=texto, bg=PANEL, fg=ACCENT,
        font=("Consolas", 8, "bold"), anchor="w"
    ).pack(fill="x", padx=10, pady=(8, 2))

def linea_separadora(parent):
    """Línea horizontal fina como separador visual."""
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=10, pady=4)


# =============================================================================
# FILA DE ARCHIVO
# Cada archivo cargado genera una instancia de esta clase.
# Contiene todos sus controles: toggle crudo/suav, color, columnas, parámetros SG.
# =============================================================================
class FilaArchivo(tk.Frame):
    def __init__(self, parent, idx, ruta, app, **kw):
        super().__init__(parent, bg=PANEL2, relief="flat", **kw)
        self.app  = app    # referencia a la App principal para llamar _redibujar()
        self.idx  = idx    # índice de esta serie (0, 1, 2 ...)
        self.ruta = ruta   # ruta completa al archivo

        # ── Colores por defecto (uno para crudo, otro para suavizado) ─────────
        ci = idx * 2
        self.color_crudo = COLORS[ci       % len(COLORS)]
        self.color_suav  = COLORS[(ci + 1) % len(COLORS)]

        # ── Variables de configuración (ligadas a los widgets Entry) ──────────
        self.var_label   = tk.StringVar(value=os.path.basename(ruta))
        self.var_col_x   = tk.IntVar(value=0)      # columna X en el archivo
        self.var_col_y   = tk.IntVar(value=1)      # columna Y en el archivo
        self.var_saltear = tk.IntVar(value=0)      # líneas a saltear al leer
        self.var_ventana = tk.IntVar(value=51)     # ventana SG (debe ser impar)
        self.var_orden   = tk.IntVar(value=3)      # orden del polinomio SG
        self.var_grosor  = tk.DoubleVar(value=1.5) # grosor de línea

        # ── Estado de visibilidad (True = visible en el gráfico) ──────────────
        self.mostrar_crudo = True
        self.mostrar_suav  = True

        # ── Panel expandible (oculto por defecto) ─────────────────────────────
        self.expandido = False

        self._construir()

    # -------------------------------------------------------------------------
    # Construcción de la fila (cabecera + cuerpo expandible)
    # -------------------------------------------------------------------------
    def _construir(self):
        # línea separadora arriba de cada fila
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # — CABECERA —
        cabecera = tk.Frame(self, bg=PANEL2)
        cabecera.pack(fill="x", padx=4, pady=(4, 2))

        # botón ▶/▼ para expandir/colapsar opciones avanzadas
        self.btn_exp = tk.Button(
            cabecera, text="▶", width=2,
            bg=PANEL2, fg=ACCENT, font=("Consolas", 8),
            relief="flat", cursor="hand2", bd=0,
            command=self._toggle_expandir
        )
        self.btn_exp.pack(side="left")

        # cuadrito de color de la línea cruda (clic = abrir selector de color)
        self.btn_color_crudo = tk.Button(
            cabecera, bg=self.color_crudo, width=2, height=1,
            relief="flat", cursor="hand2",
            command=lambda: self._elegir_color("crudo")
        )
        self.btn_color_crudo.pack(side="left", padx=(3, 0))

        # cuadrito de color de la línea suavizada
        self.btn_color_suav = tk.Button(
            cabecera, bg=self.color_suav, width=2, height=1,
            relief="flat", cursor="hand2",
            command=lambda: self._elegir_color("suav")
        )
        self.btn_color_suav.pack(side="left", padx=(2, 5))

        # nombre editable de la serie
        tk.Entry(
            cabecera, textvariable=self.var_label,
            bg=PANEL2, fg=TEXT, insertbackground=ACCENT,
            font=("Consolas", 8), relief="flat", bd=0
        ).pack(side="left", fill="x", expand=True)

        # — BOTONES TOGGLE (alineados a la derecha) —

        # botón para mostrar/ocultar la línea suavizada
        self.btn_suav = tk.Button(
            cabecera, text="SUAV ●",
            bg="#3a2010", fg=ACCENT2,
            font=("Consolas", 7, "bold"),
            relief="flat", cursor="hand2", bd=0, padx=5,
            command=self._toggle_suav
        )
        self.btn_suav.pack(side="right", padx=(2, 0))

        # botón para mostrar/ocultar la línea cruda
        self.btn_crudo = tk.Button(
            cabecera, text="CRUDO ●",
            bg="#1e3a5f", fg=ACCENT,
            font=("Consolas", 7, "bold"),
            relief="flat", cursor="hand2", bd=0, padx=5,
            command=self._toggle_crudo
        )
        self.btn_crudo.pack(side="right", padx=(4, 2))

        # botón eliminar esta serie
        tk.Button(
            cabecera, text="✕",
            bg=PANEL2, fg=DANGER,
            font=("Consolas", 10, "bold"),
            relief="flat", cursor="hand2", bd=0, padx=4,
            command=self._eliminar
        ).pack(side="right", padx=(6, 2))

        # — CUERPO EXPANDIBLE (oculto hasta que el usuario haga clic en ▶) —
        self.cuerpo = tk.Frame(self, bg=PANEL2)
        # (no se empaca aquí — se empaca/desempaca en _toggle_expandir)

        # fila 1: columnas y líneas a saltear
        fila1 = tk.Frame(self.cuerpo, bg=PANEL2)
        fila1.pack(fill="x", padx=12, pady=(2, 0))
        for etiqueta, var, ancho in [
            ("Col X:", self.var_col_x,   3),
            ("Col Y:", self.var_col_y,   3),
            ("Skip:",  self.var_saltear, 3),
        ]:
            tk.Label(fila1, text=etiqueta, bg=PANEL2, fg=SUBTEXT,
                     font=("Consolas", 8)).pack(side="left")
            hacer_entry(fila1, var, ancho)

        # fila 2: parámetros de Savitzky-Golay y grosor
        fila2 = tk.Frame(self.cuerpo, bg=PANEL2)
        fila2.pack(fill="x", padx=12, pady=(2, 4))
        for etiqueta, var, ancho in [
            ("Ventana:", self.var_ventana, 5),
            ("Orden:",   self.var_orden,   3),
            ("Grosor:",  self.var_grosor,  4),
        ]:
            tk.Label(fila2, text=etiqueta, bg=PANEL2, fg=SUBTEXT,
                     font=("Consolas", 8)).pack(side="left")
            hacer_entry(fila2, var, ancho)

    # -------------------------------------------------------------------------
    # Toggle expandir/colapsar opciones avanzadas
    # -------------------------------------------------------------------------
    def _toggle_expandir(self):
        self.expandido = not self.expandido
        if self.expandido:
            self.cuerpo.pack(fill="x")
            self.btn_exp.config(text="▼")
        else:
            self.cuerpo.pack_forget()
            self.btn_exp.config(text="▶")

    # -------------------------------------------------------------------------
    # Toggle visibilidad de la línea CRUDA
    # -------------------------------------------------------------------------
    def _toggle_crudo(self):
        self.mostrar_crudo = not self.mostrar_crudo
        if self.mostrar_crudo:
            # activo: fondo azul oscuro, texto azul, círculo relleno
            self.btn_crudo.config(text="CRUDO ●", bg="#1e3a5f", fg=ACCENT)
        else:
            # inactivo: fondo neutro, texto gris, círculo vacío
            self.btn_crudo.config(text="CRUDO ○", bg=PANEL2, fg=SUBTEXT)
        # redibujar el gráfico inmediatamente
        self.app._redibujar()

    # -------------------------------------------------------------------------
    # Toggle visibilidad de la línea SUAVIZADA
    # -------------------------------------------------------------------------
    def _toggle_suav(self):
        self.mostrar_suav = not self.mostrar_suav
        if self.mostrar_suav:
            self.btn_suav.config(text="SUAV ●", bg="#3a2010", fg=ACCENT2)
        else:
            self.btn_suav.config(text="SUAV ○", bg=PANEL2, fg=SUBTEXT)
        # redibujar el gráfico inmediatamente
        self.app._redibujar()

    # -------------------------------------------------------------------------
    # Selector de color (abre el diálogo de color del sistema)
    # -------------------------------------------------------------------------
    def _elegir_color(self, cual):
        color_actual = self.color_crudo if cual == "crudo" else self.color_suav
        _, hex_color = colorchooser.askcolor(color=color_actual, title="Elegir color")
        if hex_color:
            if cual == "crudo":
                self.color_crudo = hex_color
                self.btn_color_crudo.config(bg=hex_color)
            else:
                self.color_suav = hex_color
                self.btn_color_suav.config(bg=hex_color)

    # -------------------------------------------------------------------------
    # Eliminar esta fila de la lista
    # -------------------------------------------------------------------------
    def _eliminar(self):
        self.app.eliminar_serie(self.idx)

    # -------------------------------------------------------------------------
    # Leer datos crudos del archivo
    # -------------------------------------------------------------------------
    def datos_crudos(self):
        """Devuelve (x, y) como arrays. Lanza ValueError si hay error."""
        return leer_archivo(
            self.ruta,
            col_x=self.var_col_x.get(),
            col_y=self.var_col_y.get(),
            saltear=self.var_saltear.get()
        )

    # -------------------------------------------------------------------------
    # Calcular suavizado Savitzky-Golay sobre y
    # -------------------------------------------------------------------------
    def datos_suavizados(self, y):
        """
        Recibe y crudo y devuelve y suavizado.
        Ajusta ventana a impar si es par.
        Lanza ValueError si los parámetros son inválidos.
        """
        ventana = self.var_ventana.get()
        orden   = self.var_orden.get()

        # la ventana SG debe ser impar
        if ventana % 2 == 0:
            ventana += 1
            self.var_ventana.set(ventana)

        if orden >= ventana:
            raise ValueError(f"Orden ({orden}) debe ser < ventana ({ventana})")
        if len(y) < ventana:
            raise ValueError(f"Ventana ({ventana}) > puntos disponibles ({len(y)})")

        return savgol_filter(y, window_length=ventana, polyorder=orden)


# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graficador Multi-Plot · Savitzky-Golay")
        self.geometry("1280x800")
        self.configure(bg=BG)
        self.resizable(True, True)

        # lista de FilaArchivo activas
        self.series = []

        # variables del panel de ejes
        self.var_xmin    = tk.StringVar(value="auto")
        self.var_xmax    = tk.StringVar(value="auto")
        self.var_ymin    = tk.StringVar(value="auto")
        self.var_ymax    = tk.StringVar(value="auto")
        self.var_xlabel  = tk.StringVar(value="X")
        self.var_ylabel  = tk.StringVar(value="Y")
        self.var_titulo  = tk.StringVar(value="")
        self.var_grilla  = tk.BooleanVar(value=True)
        self.var_leyenda = tk.BooleanVar(value=True)

        self._construir_ui()

    # =========================================================================
    # CONSTRUCCIÓN DE LA INTERFAZ
    # =========================================================================
    def _construir_ui(self):
        # dos columnas: panel izquierdo fijo (300px) + gráfico expansible
        self.columnconfigure(0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # — PANEL IZQUIERDO —
        panel_izq = tk.Frame(self, bg=PANEL, width=300)
        panel_izq.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        panel_izq.grid_propagate(False)
        panel_izq.rowconfigure(1, weight=1)
        panel_izq.columnconfigure(0, weight=1)

        tk.Label(
            panel_izq, text="  PLOTS  &  EJES",
            bg=PANEL, fg=ACCENT,
            font=("Consolas", 11, "bold"), anchor="w"
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=(10, 4))

        # — NOTEBOOK (pestañas Archivos / Ejes) —
        estilo = ttk.Style()
        estilo.theme_use("default")
        estilo.configure("D.TNotebook",     background=PANEL, borderwidth=0)
        estilo.configure("D.TNotebook.Tab", background=BORDER, foreground=SUBTEXT,
                         font=("Consolas", 8, "bold"), padding=[10, 4])
        estilo.map("D.TNotebook.Tab",
                   background=[("selected", BG)],
                   foreground=[("selected", ACCENT)])

        nb = ttk.Notebook(panel_izq, style="D.TNotebook")
        nb.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))

        tab_archivos = tk.Frame(nb, bg=PANEL)
        nb.add(tab_archivos, text="  Archivos  ")
        self._construir_tab_archivos(tab_archivos)

        tab_ejes = tk.Frame(nb, bg=PANEL)
        nb.add(tab_ejes, text="  Ejes  ")
        self._construir_tab_ejes(tab_ejes)

        # — BOTÓN GRAFICAR —
        tk.Button(
            panel_izq, text="▶   GRAFICAR TODO",
            bg=ACCENT, fg="#fff", font=("Consolas", 10, "bold"),
            relief="flat", cursor="hand2", pady=9,
            activebackground="#3a7fd8",
            command=self._redibujar
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 3))

        # etiqueta de estado (muestra cuántas series y puntos)
        self.lbl_status = tk.Label(
            panel_izq, text="", bg=PANEL, fg=SUCCESS,
            font=("Consolas", 8), wraplength=280, justify="left"
        )
        self.lbl_status.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))

        # — PANEL DERECHO (gráfico) —
        panel_der = tk.Frame(self, bg=BG)
        panel_der.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        panel_der.rowconfigure(0, weight=1)
        panel_der.columnconfigure(0, weight=1)
        self._construir_grafico(panel_der)

    # -------------------------------------------------------------------------
    # Pestaña: lista de archivos cargados
    # -------------------------------------------------------------------------
    def _construir_tab_archivos(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # botones de acción
        frame_botones = tk.Frame(parent, bg=PANEL)
        frame_botones.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        tk.Button(
            frame_botones, text="＋  Agregar archivo(s)",
            bg=ACCENT, fg="#fff", font=("Consolas", 9, "bold"),
            relief="flat", cursor="hand2", pady=5,
            command=self._agregar_archivos
        ).pack(fill="x")

        tk.Button(
            frame_botones, text="✕  Limpiar todo",
            bg=BORDER, fg=DANGER, font=("Consolas", 8),
            relief="flat", cursor="hand2", pady=3,
            command=self._limpiar_todo
        ).pack(fill="x", pady=(4, 0))

        # área scrolleable que contiene las FilaArchivo
        contenedor = tk.Frame(parent, bg=PANEL)
        contenedor.grid(row=1, column=0, sticky="nsew", padx=2)
        contenedor.rowconfigure(0, weight=1)
        contenedor.columnconfigure(0, weight=1)

        self.canvas_lista = tk.Canvas(contenedor, bg=PANEL, highlightthickness=0)
        self.canvas_lista.grid(row=0, column=0, sticky="nsew")

        scroll = tk.Scrollbar(contenedor, orient="vertical",
                              command=self.canvas_lista.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.canvas_lista.configure(yscrollcommand=scroll.set)

        # frame interno donde se agregan las filas
        self.frame_lista = tk.Frame(self.canvas_lista, bg=PANEL)
        self._id_frame = self.canvas_lista.create_window(
            (0, 0), window=self.frame_lista, anchor="nw"
        )

        def _actualizar_scroll(e):
            self.canvas_lista.configure(scrollregion=self.canvas_lista.bbox("all"))
            self.canvas_lista.itemconfig(self._id_frame,
                                         width=self.canvas_lista.winfo_width())

        self.frame_lista.bind("<Configure>", _actualizar_scroll)
        self.canvas_lista.bind("<Configure>", _actualizar_scroll)
        # scroll con rueda del mouse
        self.canvas_lista.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas_lista.yview_scroll(int(-1*(e.delta/120)), "units")
        )

    # -------------------------------------------------------------------------
    # Pestaña: controles de ejes
    # -------------------------------------------------------------------------
    def _construir_tab_ejes(self, parent):
        def fila_entrada(p, etiqueta, var, hint=""):
            """Crea una fila con etiqueta + campo de entrada."""
            f = tk.Frame(p, bg=PANEL)
            f.pack(fill="x", padx=10, pady=2)
            tk.Label(f, text=etiqueta, bg=PANEL, fg=SUBTEXT,
                     font=("Consolas", 8), width=9, anchor="w").pack(side="left")
            tk.Entry(f, textvariable=var, bg=BORDER, fg=TEXT,
                     insertbackground=ACCENT, font=("Consolas", 9),
                     relief="flat", bd=0, width=10).pack(side="left", ipady=3)
            if hint:
                tk.Label(f, text=hint, bg=PANEL, fg=BORDER,
                         font=("Consolas", 7)).pack(side="left", padx=4)

        etiqueta_seccion(parent, "Rango X")
        fila_entrada(parent, "X min:", self.var_xmin, "auto")
        fila_entrada(parent, "X max:", self.var_xmax, "auto")

        etiqueta_seccion(parent, "Rango Y")
        fila_entrada(parent, "Y min:", self.var_ymin, "auto")
        fila_entrada(parent, "Y max:", self.var_ymax, "auto")

        linea_separadora(parent)
        etiqueta_seccion(parent, "Etiquetas")
        fila_entrada(parent, "Titulo:", self.var_titulo)
        fila_entrada(parent, "Eje X:",  self.var_xlabel)
        fila_entrada(parent, "Eje Y:",  self.var_ylabel)

        linea_separadora(parent)
        etiqueta_seccion(parent, "Opciones")
        for texto, var in [("Mostrar grilla",  self.var_grilla),
                           ("Mostrar leyenda", self.var_leyenda)]:
            tk.Checkbutton(
                parent, text=texto, variable=var,
                bg=PANEL, fg=TEXT, selectcolor=BG,
                activebackground=PANEL, font=("Consolas", 9)
            ).pack(anchor="w", padx=14)

        linea_separadora(parent)
        tk.Button(
            parent, text="Aplicar ejes",
            bg=BORDER, fg=ACCENT, font=("Consolas", 9),
            relief="flat", cursor="hand2", pady=5,
            command=self._aplicar_ejes
        ).pack(fill="x", padx=10, pady=4)

    # -------------------------------------------------------------------------
    # Área del gráfico matplotlib
    # -------------------------------------------------------------------------
    def _construir_grafico(self, parent):
        self.fig = Figure(facecolor=BG, edgecolor=BG)
        self.ax  = self.fig.add_subplot(111)
        self._estilo_ejes()

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_fig.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # barra de herramientas matplotlib (zoom, pan, guardar)
        frame_tb = tk.Frame(parent, bg=PANEL)
        frame_tb.grid(row=1, column=0, sticky="ew")
        tb = NavigationToolbar2Tk(self.canvas_fig, frame_tb)
        tb.config(bg=PANEL)
        tb.update()

    def _estilo_ejes(self):
        """Aplica el tema oscuro a los ejes matplotlib."""
        ax = self.ax
        ax.set_facecolor("#12151e")
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

    # =========================================================================
    # ACCIONES
    # =========================================================================

    def _agregar_archivos(self):
        """Abre el diálogo de selección y agrega cada archivo como nueva fila."""
        rutas = filedialog.askopenfilenames(
            title="Seleccionar archivos",
            filetypes=[("Texto/CSV/DAT", "*.txt *.csv *.dat"), ("Todos", "*.*")]
        )
        for ruta in rutas:
            idx  = len(self.series)
            fila = FilaArchivo(self.frame_lista, idx, ruta, self)
            fila.pack(fill="x", padx=2, pady=1)
            self.series.append(fila)

    def eliminar_serie(self, idx):
        """Elimina la fila con el índice dado y re-indexa las demás."""
        for fila in self.series:
            if fila.idx == idx:
                fila.destroy()
                self.series.remove(fila)
                break
        # re-indexar para mantener idx correcto
        for i, fila in enumerate(self.series):
            fila.idx = i
        # redibujar sin la serie eliminada
        self._redibujar()

    def _limpiar_todo(self):
        """Elimina todas las series y limpia el gráfico."""
        for fila in self.series:
            fila.destroy()
        self.series.clear()
        self.ax.clear()
        self._estilo_ejes()
        self.canvas_fig.draw()
        self.lbl_status.config(text="")

    # -------------------------------------------------------------------------
    # Redibujar el gráfico completo
    # Llamado por: botón "Graficar todo", toggles crudo/suav, eliminar serie
    # -------------------------------------------------------------------------
    def _redibujar(self):
        # si no hay series simplemente limpiar y salir — sin messagebox
        if not self.series:
            self.ax.clear()
            self._estilo_ejes()
            self.canvas_fig.draw()
            return

        self.ax.clear()
        self._estilo_ejes()

        errores   = []
        total_pts = 0

        for fila in self.series:
            nombre = fila.var_label.get()
            grosor = fila.var_grosor.get()

            # — leer datos crudos —
            try:
                x, y = fila.datos_crudos()
            except Exception as e:
                errores.append(f"[{nombre}] Error leyendo: {e}")
                continue

            total_pts += len(x)

            # — dibujar línea cruda (solo si mostrar_crudo es True) —
            if fila.mostrar_crudo:
                self.ax.plot(
                    x, y,
                    color=fila.color_crudo,
                    linewidth=max(grosor * 0.5, 0.5),
                    alpha=0.45,
                    label=f"{nombre} · crudo",
                    zorder=2
                )

            # — calcular y dibujar suavizado (solo si mostrar_suav es True) —
            if fila.mostrar_suav:
                try:
                    y_suav  = fila.datos_suavizados(y)
                    ventana = fila.var_ventana.get()
                    orden   = fila.var_orden.get()
                    self.ax.plot(
                        x, y_suav,
                        color=fila.color_suav,
                        linewidth=grosor,
                        label=f"{nombre} · SG(w={ventana},o={orden})",
                        zorder=3
                    )
                except Exception as e:
                    errores.append(f"[{nombre}] Suavizado: {e}")

        # aplicar configuración de ejes antes de redibujar
        self._aplicar_ejes(redibujar=False)

        if errores:
            messagebox.showwarning("Advertencias", "\n".join(errores))

        self.lbl_status.config(
            text=f"✓  {len(self.series)} serie(s)  ·  {total_pts} pts totales"
        )
        self.canvas_fig.draw()

    # -------------------------------------------------------------------------
    # Aplicar configuración de ejes (rango, etiquetas, grilla, leyenda)
    # -------------------------------------------------------------------------
    def _aplicar_ejes(self, redibujar=True):
        ax = self.ax

        # rango X — si el campo no es número válido se deja en "auto"
        try:    xmin = float(self.var_xmin.get())
        except: xmin = None
        try:    xmax = float(self.var_xmax.get())
        except: xmax = None
        if xmin is not None or xmax is not None:
            cur = ax.get_xlim()
            ax.set_xlim(
                xmin if xmin is not None else cur[0],
                xmax if xmax is not None else cur[1]
            )

        # rango Y
        try:    ymin = float(self.var_ymin.get())
        except: ymin = None
        try:    ymax = float(self.var_ymax.get())
        except: ymax = None
        if ymin is not None or ymax is not None:
            cur = ax.get_ylim()
            ax.set_ylim(
                ymin if ymin is not None else cur[0],
                ymax if ymax is not None else cur[1]
            )

        # etiquetas
        titulo = self.var_titulo.get().strip()
        ax.set_title(titulo if titulo else "", color=TEXT, fontsize=11, pad=10)
        ax.set_xlabel(self.var_xlabel.get(), fontsize=10, labelpad=6)
        ax.set_ylabel(self.var_ylabel.get(), fontsize=10, labelpad=6)

        # grilla
        ax.grid(self.var_grilla.get(), color=BORDER,
                linestyle="--", linewidth=0.5, alpha=0.7)

        # leyenda
        handles, labels = ax.get_legend_handles_labels()
        if self.var_leyenda.get() and handles:
            ax.legend(facecolor=PANEL, edgecolor=BORDER,
                      labelcolor=TEXT, fontsize=8,
                      loc="best", framealpha=0.9)
        elif ax.get_legend():
            ax.get_legend().remove()

        self.fig.tight_layout()
        if redibujar:
            self.canvas_fig.draw()


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    # instalar dependencias si faltan
    try:
        import scipy, matplotlib
    except ImportError:
        import subprocess, sys
        subprocess.check_call([
            sys.executable, "-m", "pip",
            "install", "scipy", "matplotlib", "numpy",
            "--break-system-packages"
        ])
    App().mainloop()