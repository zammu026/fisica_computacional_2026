"""
Graficador Multi-Plot con suavizado Savitzky-Golay
Uso: python grafico_savgol.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
from scipy.signal import savgol_filter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os

# ── Paleta ────────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
PANEL2  = "#141720"
ACCENT  = "#4f9cf9"
ACCENT2 = "#f97b4f"
TEXT    = "#e8eaf0"
SUBTEXT = "#7b8099"
BORDER  = "#2a2d3e"
SUCCESS = "#4fcf80"
DANGER  = "#f94f4f"
COLORS  = ["#4f9cf9","#f97b4f","#4fcf80","#f9c84f","#c44ff9",
           "#4ff9f0","#f94f9c","#9cf94f","#f96060","#4f6af9"]

# ── Lectura de archivo ────────────────────────────────────────────────────────
def leer_txt(ruta, col_x=0, col_y=1, saltear=0):
    datos = []
    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        lineas = f.readlines()
    for i, linea in enumerate(lineas):
        if i < saltear:
            continue
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
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
            continue
    if not datos:
        raise ValueError("No se encontraron datos numéricos.")
    arr = np.array(datos)
    return arr[:, col_x], arr[:, col_y]

# ── Helpers UI ────────────────────────────────────────────────────────────────
def make_entry(parent, var, width=6):
    e = tk.Entry(parent, textvariable=var, bg=BORDER, fg=TEXT,
                 insertbackground=ACCENT, font=("Consolas", 9),
                 relief="flat", bd=0, width=width)
    e.pack(side="left", ipady=3, padx=(0, 3))
    return e

def section_label(parent, text):
    tk.Label(parent, text=text, bg=PANEL, fg=ACCENT,
             font=("Consolas", 8, "bold"), anchor="w"
             ).pack(fill="x", padx=10, pady=(8, 2))

def sep_line(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=10, pady=4)

# ── Fila de un plot ───────────────────────────────────────────────────────────
class PlotRow(tk.Frame):
    def __init__(self, parent, idx, ruta, app, **kw):
        super().__init__(parent, bg=PANEL2, relief="flat", **kw)
        self.app  = app
        self.idx  = idx
        self.ruta = ruta

        # colores por defecto — uno por serie
        ci = idx * 2
        self.color_crudo = COLORS[ci       % len(COLORS)]
        self.color_suav  = COLORS[(ci + 1) % len(COLORS)]

        # variables de configuración
        self.var_label   = tk.StringVar(value=os.path.basename(ruta))
        self.var_col_x   = tk.IntVar(value=0)
        self.var_col_y   = tk.IntVar(value=1)
        self.var_saltear = tk.IntVar(value=0)
        self.var_ventana = tk.IntVar(value=51)
        self.var_orden   = tk.IntVar(value=3)
        self.var_grosor  = tk.DoubleVar(value=1.5)

        # visibilidad — True = visible
        self.mostrar_crudo = True
        self.mostrar_suav  = True

        # expandido
        self.expanded = False

        self._build()

    # ── construcción de la fila ───────────────────────────────────────────────
    def _build(self):
        # separador superior
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # cabecera
        hdr = tk.Frame(self, bg=PANEL2)
        hdr.pack(fill="x", padx=4, pady=(4, 2))

        # flecha expand
        self.btn_exp = tk.Button(
            hdr, text="▶", width=2,
            bg=PANEL2, fg=ACCENT, font=("Consolas", 8),
            relief="flat", cursor="hand2", bd=0,
            command=self._toggle_expand)
        self.btn_exp.pack(side="left")

        # pastilla color crudo
        self.btn_cc = tk.Button(
            hdr, bg=self.color_crudo, width=2, height=1,
            relief="flat", cursor="hand2",
            command=lambda: self._pick_color("crudo"))
        self.btn_cc.pack(side="left", padx=(3, 0))

        # pastilla color suav
        self.btn_cs = tk.Button(
            hdr, bg=self.color_suav, width=2, height=1,
            relief="flat", cursor="hand2",
            command=lambda: self._pick_color("suav"))
        self.btn_cs.pack(side="left", padx=(2, 5))

        # nombre editable
        tk.Entry(hdr, textvariable=self.var_label,
                 bg=PANEL2, fg=TEXT, insertbackground=ACCENT,
                 font=("Consolas", 8), relief="flat", bd=0
                 ).pack(side="left", fill="x", expand=True)

        # ── botones toggle Crudo / Suavizado ──────────────────────────────────
        self.btn_tog_s = tk.Button(
            hdr, text="SUAV ●",
            bg="#3a2010", fg=ACCENT2,
            font=("Consolas", 7, "bold"),
            relief="flat", cursor="hand2", bd=0, padx=5,
            command=self._click_suav)
        self.btn_tog_s.pack(side="right", padx=(2, 0))

        self.btn_tog_c = tk.Button(
            hdr, text="CRUDO ●",
            bg="#1e3a5f", fg=ACCENT,
            font=("Consolas", 7, "bold"),
            relief="flat", cursor="hand2", bd=0, padx=5,
            command=self._click_crudo)
        self.btn_tog_c.pack(side="right", padx=(4, 2))

        # botón eliminar esta fila
        self.btn_del = tk.Button(
            hdr, text="✕",
            bg=PANEL2, fg=DANGER,
            font=("Consolas", 10, "bold"),
            relief="flat", cursor="hand2", bd=0, padx=4,
            command=self._remove)
        self.btn_del.pack(side="right", padx=(6, 2))

        # ── cuerpo expandible ─────────────────────────────────────────────────
        self.body = tk.Frame(self, bg=PANEL2)

        row1 = tk.Frame(self.body, bg=PANEL2)
        row1.pack(fill="x", padx=12, pady=(2, 0))
        for ltext, var, w in [("Col X:", self.var_col_x, 3),
                               ("Col Y:", self.var_col_y, 3),
                               ("Skip:",  self.var_saltear, 3)]:
            tk.Label(row1, text=ltext, bg=PANEL2, fg=SUBTEXT,
                     font=("Consolas", 8)).pack(side="left")
            make_entry(row1, var, w)

        row2 = tk.Frame(self.body, bg=PANEL2)
        row2.pack(fill="x", padx=12, pady=(2, 4))
        for ltext, var, w in [("Ventana:", self.var_ventana, 5),
                               ("Orden:",  self.var_orden,   3),
                               ("Grosor:", self.var_grosor,  4)]:
            tk.Label(row2, text=ltext, bg=PANEL2, fg=SUBTEXT,
                     font=("Consolas", 8)).pack(side="left")
            make_entry(row2, var, w)

    # ── toggle expand ─────────────────────────────────────────────────────────
    def _toggle_expand(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.body.pack(fill="x")
            self.btn_exp.config(text="▼")
        else:
            self.body.pack_forget()
            self.btn_exp.config(text="▶")

    # ── toggle crudo ─────────────────────────────────────────────────────────
    def _click_crudo(self):
        self.mostrar_crudo = not self.mostrar_crudo
        if self.mostrar_crudo:
            self.btn_tog_c.config(text="CRUDO ●", bg="#1e3a5f", fg=ACCENT)
        else:
            self.btn_tog_c.config(text="CRUDO ○", bg=PANEL2,    fg=SUBTEXT)
        # Redibujar inmediatamente sin esperar "Graficar todo"
        self.app._graficar()

    # ── toggle suavizado ──────────────────────────────────────────────────────
    def _click_suav(self):
        self.mostrar_suav = not self.mostrar_suav
        if self.mostrar_suav:
            self.btn_tog_s.config(text="SUAV ●", bg="#3a2010", fg=ACCENT2)
        else:
            self.btn_tog_s.config(text="SUAV ○", bg=PANEL2,    fg=SUBTEXT)
        # Redibujar inmediatamente sin esperar "Graficar todo"
        self.app._graficar()

    # ── selector de color ────────────────────────────────────────────────────
    def _pick_color(self, which):
        ini = self.color_crudo if which == "crudo" else self.color_suav
        _, hex_c = colorchooser.askcolor(color=ini, title="Elegir color")
        if hex_c:
            if which == "crudo":
                self.color_crudo = hex_c
                self.btn_cc.config(bg=hex_c)
            else:
                self.color_suav = hex_c
                self.btn_cs.config(bg=hex_c)

    # ── eliminar esta fila ────────────────────────────────────────────────────
    def _remove(self):
        self.app.remove_plot(self.idx)

    # ── obtener datos crudos ──────────────────────────────────────────────────
    def get_xy_crudo(self):
        """Devuelve (x, y) crudos. Lanza ValueError si falla."""
        return leer_txt(self.ruta,
                        col_x=self.var_col_x.get(),
                        col_y=self.var_col_y.get(),
                        saltear=self.var_saltear.get())

    def get_xy_suav(self, y):
        """Devuelve y_suavizado dado y crudo. Lanza ValueError si falla."""
        w = self.var_ventana.get()
        o = self.var_orden.get()
        if w % 2 == 0:
            w += 1
            self.var_ventana.set(w)
        if o >= w:
            raise ValueError(f"Orden ({o}) debe ser menor que ventana ({w})")
        if len(y) < w:
            raise ValueError(f"Ventana ({w}) mayor que puntos ({len(y)})")
        return savgol_filter(y, window_length=w, polyorder=o)


# ── Aplicación principal ──────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graficador Multi-Plot · Savitzky-Golay")
        self.geometry("1280x800")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.plot_rows = []

        self.var_xmin   = tk.StringVar(value="auto")
        self.var_xmax   = tk.StringVar(value="auto")
        self.var_ymin   = tk.StringVar(value="auto")
        self.var_ymax   = tk.StringVar(value="auto")
        self.var_xlabel = tk.StringVar(value="X")
        self.var_ylabel = tk.StringVar(value="Y")
        self.var_title  = tk.StringVar(value="")
        self.var_grid   = tk.BooleanVar(value=True)
        self.var_legend = tk.BooleanVar(value=True)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = tk.Frame(self, bg=PANEL, width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        left.grid_propagate(False)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        tk.Label(left, text="  PLOTS  &  EJES", bg=PANEL, fg=ACCENT,
                 font=("Consolas", 11, "bold"), anchor="w"
                 ).grid(row=0, column=0, sticky="ew", padx=6, pady=(10, 4))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("D.TNotebook",     background=PANEL, borderwidth=0)
        style.configure("D.TNotebook.Tab", background=BORDER, foreground=SUBTEXT,
                        font=("Consolas", 8, "bold"), padding=[10, 4])
        style.map("D.TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])

        nb = ttk.Notebook(left, style="D.TNotebook")
        nb.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))

        t_files = tk.Frame(nb, bg=PANEL)
        nb.add(t_files, text="  Archivos  ")
        self._build_tab_files(t_files)

        t_axes = tk.Frame(nb, bg=PANEL)
        nb.add(t_axes, text="  Ejes  ")
        self._build_tab_axes(t_axes)

        tk.Button(left, text="▶   GRAFICAR TODO",
                  bg=ACCENT, fg="#fff", font=("Consolas", 10, "bold"),
                  relief="flat", cursor="hand2", pady=9,
                  activebackground="#3a7fd8",
                  command=self._graficar
                  ).grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 3))

        self.lbl_status = tk.Label(
            left, text="", bg=PANEL, fg=SUCCESS,
            font=("Consolas", 8), wraplength=280, justify="left")
        self.lbl_status.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))

        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_chart(right)

    # ── tab archivos ──────────────────────────────────────────────────────────
    def _build_tab_files(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        btn_f = tk.Frame(parent, bg=PANEL)
        btn_f.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        tk.Button(btn_f, text="＋  Agregar archivo(s)",
                  bg=ACCENT, fg="#fff", font=("Consolas", 9, "bold"),
                  relief="flat", cursor="hand2", pady=5,
                  command=self._agregar_archivos
                  ).pack(fill="x")

        tk.Button(btn_f, text="✕  Limpiar todo",
                  bg=BORDER, fg=DANGER, font=("Consolas", 8),
                  relief="flat", cursor="hand2", pady=3,
                  command=self._limpiar_todo
                  ).pack(fill="x", pady=(4, 0))

        cont = tk.Frame(parent, bg=PANEL)
        cont.grid(row=1, column=0, sticky="nsew", padx=2)
        cont.rowconfigure(0, weight=1)
        cont.columnconfigure(0, weight=1)

        self._cs = tk.Canvas(cont, bg=PANEL, highlightthickness=0)
        self._cs.grid(row=0, column=0, sticky="nsew")

        sb = tk.Scrollbar(cont, orient="vertical", command=self._cs.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._cs.configure(yscrollcommand=sb.set)

        self.list_frame = tk.Frame(self._cs, bg=PANEL)
        self._lf_id = self._cs.create_window(
            (0, 0), window=self.list_frame, anchor="nw")

        def _cfg(e):
            self._cs.configure(scrollregion=self._cs.bbox("all"))
            self._cs.itemconfig(self._lf_id, width=self._cs.winfo_width())

        self.list_frame.bind("<Configure>", _cfg)
        self._cs.bind("<Configure>", _cfg)
        self._cs.bind_all("<MouseWheel>",
            lambda e: self._cs.yview_scroll(int(-1*(e.delta/120)), "units"))

    # ── tab ejes ──────────────────────────────────────────────────────────────
    def _build_tab_axes(self, parent):
        def row_entry(p, label, var, hint=""):
            f = tk.Frame(p, bg=PANEL)
            f.pack(fill="x", padx=10, pady=2)
            tk.Label(f, text=label, bg=PANEL, fg=SUBTEXT,
                     font=("Consolas", 8), width=9, anchor="w").pack(side="left")
            tk.Entry(f, textvariable=var, bg=BORDER, fg=TEXT,
                     insertbackground=ACCENT, font=("Consolas", 9),
                     relief="flat", bd=0, width=10).pack(side="left", ipady=3)
            if hint:
                tk.Label(f, text=hint, bg=PANEL, fg=BORDER,
                         font=("Consolas", 7)).pack(side="left", padx=4)

        section_label(parent, "Rango X")
        row_entry(parent, "X min:", self.var_xmin, "auto")
        row_entry(parent, "X max:", self.var_xmax, "auto")

        section_label(parent, "Rango Y")
        row_entry(parent, "Y min:", self.var_ymin, "auto")
        row_entry(parent, "Y max:", self.var_ymax, "auto")

        sep_line(parent)
        section_label(parent, "Etiquetas")
        row_entry(parent, "Titulo:", self.var_title)
        row_entry(parent, "Eje X:",  self.var_xlabel)
        row_entry(parent, "Eje Y:",  self.var_ylabel)

        sep_line(parent)
        section_label(parent, "Opciones")

        for text, var in [("Mostrar grilla",  self.var_grid),
                          ("Mostrar leyenda", self.var_legend)]:
            tk.Checkbutton(parent, text=text, variable=var,
                           bg=PANEL, fg=TEXT, selectcolor=BG,
                           activebackground=PANEL,
                           font=("Consolas", 9)
                           ).pack(anchor="w", padx=14)

        sep_line(parent)
        tk.Button(parent, text="Aplicar ejes",
                  bg=BORDER, fg=ACCENT, font=("Consolas", 9),
                  relief="flat", cursor="hand2", pady=5,
                  command=self._aplicar_ejes
                  ).pack(fill="x", padx=10, pady=4)

    # ── gráfica ───────────────────────────────────────────────────────────────
    def _build_chart(self, parent):
        self.fig = Figure(facecolor=BG, edgecolor=BG)
        self.ax  = self.fig.add_subplot(111)
        self._estilo_ax()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tb_f = tk.Frame(parent, bg=PANEL)
        tb_f.grid(row=1, column=0, sticky="ew")
        tb = NavigationToolbar2Tk(self.canvas, tb_f)
        tb.config(bg=PANEL)
        tb.update()

    def _estilo_ax(self):
        ax = self.ax
        ax.set_facecolor("#12151e")
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

    # ── acciones ──────────────────────────────────────────────────────────────
    def _agregar_archivos(self):
        rutas = filedialog.askopenfilenames(
            title="Seleccionar archivos",
            filetypes=[("Texto/CSV/DAT", "*.txt *.csv *.dat"), ("Todos", "*.*")])
        for ruta in rutas:
            idx = len(self.plot_rows)
            row = PlotRow(self.list_frame, idx, ruta, self)
            row.pack(fill="x", padx=2, pady=1)
            self.plot_rows.append(row)

    def remove_plot(self, idx):
        for row in self.plot_rows:
            if row.idx == idx:
                row.destroy()
                self.plot_rows.remove(row)
                break
        for i, row in enumerate(self.plot_rows):
            row.idx = i

    def _limpiar_todo(self):
        for row in self.plot_rows:
            row.destroy()
        self.plot_rows.clear()
        self.ax.clear()
        self._estilo_ax()
        self.canvas.draw()

    # ── graficar ──────────────────────────────────────────────────────────────
    def _graficar(self):
        if not self.plot_rows:
            messagebox.showinfo("Sin datos", "Agrega al menos un archivo.")
            return

        self.ax.clear()
        self._estilo_ax()

        errores   = []
        total_pts = 0

        for row in self.plot_rows:
            label = row.var_label.get()
            lw    = row.var_grosor.get()

            try:
                x, y = row.get_xy_crudo()
            except Exception as e:
                errores.append(f"[{label}] Error leyendo: {e}")
                continue

            total_pts += len(x)

            # dibujar crudo si está activo
            if row.mostrar_crudo:
                self.ax.plot(x, y,
                             color=row.color_crudo,
                             linewidth=max(lw * 0.5, 0.5),
                             alpha=0.45,
                             label=f"{label} · crudo",
                             zorder=2)

            # calcular y dibujar suavizado si está activo
            if row.mostrar_suav:
                try:
                    y_suav = row.get_xy_suav(y)
                    w = row.var_ventana.get()
                    o = row.var_orden.get()
                    self.ax.plot(x, y_suav,
                                 color=row.color_suav,
                                 linewidth=lw,
                                 label=f"{label} · SG(w={w},o={o})",
                                 zorder=3)
                except Exception as e:
                    errores.append(f"[{label}] SG: {e}")

        self._aplicar_ejes(redraw=False)

        if errores:
            messagebox.showwarning("Advertencias", "\n".join(errores))

        self.lbl_status.config(
            text=f"✓  {len(self.plot_rows)} serie(s)  ·  {total_pts} pts totales")
        self.canvas.draw()

    def _aplicar_ejes(self, redraw=True):
        ax = self.ax

        try:    xmin = float(self.var_xmin.get())
        except: xmin = None
        try:    xmax = float(self.var_xmax.get())
        except: xmax = None
        if xmin is not None or xmax is not None:
            cur = ax.get_xlim()
            ax.set_xlim(xmin if xmin is not None else cur[0],
                        xmax if xmax is not None else cur[1])

        try:    ymin = float(self.var_ymin.get())
        except: ymin = None
        try:    ymax = float(self.var_ymax.get())
        except: ymax = None
        if ymin is not None or ymax is not None:
            cur = ax.get_ylim()
            ax.set_ylim(ymin if ymin is not None else cur[0],
                        ymax if ymax is not None else cur[1])

        t = self.var_title.get().strip()
        ax.set_title(t if t else "", color=TEXT, fontsize=11, pad=10)
        ax.set_xlabel(self.var_xlabel.get(), fontsize=10, labelpad=6)
        ax.set_ylabel(self.var_ylabel.get(), fontsize=10, labelpad=6)

        ax.grid(self.var_grid.get(), color=BORDER,
                linestyle="--", linewidth=0.5, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        if self.var_legend.get() and handles:
            ax.legend(facecolor=PANEL, edgecolor=BORDER,
                      labelcolor=TEXT, fontsize=8,
                      loc="best", framealpha=0.9)
        elif ax.get_legend():
            ax.get_legend().remove()

        self.fig.tight_layout()
        if redraw:
            self.canvas.draw()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import scipy, matplotlib
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "scipy", "matplotlib", "numpy",
                               "--break-system-packages"])
    App().mainloop()