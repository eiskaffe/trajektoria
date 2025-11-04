import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ussa1976

# Paraméterek

F_t = 400          # [kg m s^-2] = [N] A motor tolóereje
I_sp = 200         # [s] a rakéta hajtómű(vek) eredő specifikus impulzusa
k = 0.6            # [-] A rakéta alaki tényezője
D = 0.3            # [m] Rakéta átmérője
m_uzemanyag = 2  # [kg] üzemanyagtömege
m_szaraz = 4     # [kg] a rakéta 'száraztömege', üzemanyag nélkül



# Szimulációs paraméterek
dt = 0.001         # [s] lépésköz (mintavételezési) időtartam
tmax = 500         # [s] maximális vizsgált időtartam.

# Konstansok
R_F = 6378e3        # [m] Föld sugara
M_F = 5.972e24      # [kg] Föld tömege
gamma = 6.6743e-11  # [m^3 kg^-1 s^-2] gravitációs konstans
rho_0 = 1.225       # [kg/m^3] tengerszinti levegő sűrűség
H = 8.4e3           # [m] Skálamagasság
g_0 = 9.80665       # [m/s^2] Nehézségi gyorsulás

# Állapotváltozók
m = m_uzemanyag + m_szaraz
v = 0.0
h = 0.0

# Számított paraméterek
V_e = I_sp*g_0
t = 0
v_e = V_e + 0
A = ((D/2)**2)*np.pi  # [m^2] a homlokfelület nagysága
mdot = F_t / (I_sp*g_0)         # [kg/s] tömegáram

t_vals, m_vals, v_vals, h_vals, F_vals, Q_vals = [], [], [], [], [], []

def tsiolkovsky():
    return v_e * np.log((m_uzemanyag + m_szaraz) / m_szaraz)

print(f"Sanity check (Tsiolkovsky): {(voa := tsiolkovsky())}")
print(f"h_max kb @ {voa*voa / (2*g_0)}")

def rho_barometrikus(h):
    return rho_0*np.e**(-h/H)

def rho_ussa1976(z_min_m=0.0, z_max_m=30000.0, step_m=100.0):
    """
    Készít egy tömb-alapú lookup függvényt, ami a beérkező magasságot a legközelebbi rácspontokra kvantálja
    és közvetlen indexeléssel ad vissza értéket. Rendkívül gyors skalár- és vektor-hívásokhoz.
    Pontosság: nearest-neighbor (ha kell, vissza lehet adni lineáris interpolált változatot is).
    """
    z_grid = np.arange(z_min_m, z_max_m + 1e-9, step_m)
    ds = ussa1976.compute(z=z_grid, variables=["rho"])
    rho_grid = ds["rho"].values.astype(float)

    def rho_lookup(z):
        scalar_input = np.isscalar(z)
        z_arr = np.atleast_1d(z).astype(float)
        # clamp
        z_arr = np.clip(z_arr, z_grid[0], z_grid[-1])
        # compute nearest index
        idx = np.rint((z_arr - z_grid[0]) / step_m).astype(int)
        idx = np.clip(idx, 0, len(z_grid)-1)
        out = rho_grid[idx]
        if scalar_input:
            return float(out[0])
        return out

    return rho_lookup, {"z_grid": z_grid, "rho_grid": rho_grid, "step_m": step_m}
rho_q_fn, _ = rho_ussa1976(z_min_m=0.0, z_max_m=voa*voa / (2*g_0), step_m=1.0)

def F(m, h, v):
    return mdot*v_e - gamma*M_F*m / ((R_F + h)**2) - 0.5*k*A*(rho_q_fn(h))*(v**2)*np.sign(v)

print("Szimuláció start")
while t <= tmax:
    f = F(m, h, v)
    
    v += f*dt/m
    h += v*dt
    
    if h < 0 or v < -10: break
    
    if m > m_szaraz: m -= mdot*dt
    else: v_e = 0
    
    t_vals.append(t)
    m_vals.append(m)
    v_vals.append(v)
    h_vals.append(h)
    F_vals.append(f)
    Q_vals.append(0.5*rho_0*(np.e**(-h/H))*(v**2))
    
    t += dt
print("Szimuláció vége")


# -----------------
# --- ÁBRÁZOLÁS ---
# -----------------

import numpy as np
import matplotlib.pyplot as plt

# gyorsulás számítása (védelem nulla tömegre)
a_vals = [f_val / (m_val*g_0) if (m_val != 0 and m_val is not None) else 0.0 for f_val, m_val in zip(F_vals, m_vals)]

# --- Fontos események (biztonságosan, StopIteration elkerülésével) ---
idx_fuel_end = next((i for i, m in enumerate(m_vals) if m <= m_szaraz), None)
t_fuel_end = t_vals[idx_fuel_end] if idx_fuel_end is not None else None

idx_v_zero = next((i for i, v in enumerate(v_vals) if v <= 0), None)
t_v_zero = t_vals[idx_v_zero] if idx_v_zero is not None else None
if idx_v_zero is not None:
    print(f"h_max = {h_vals[idx_v_zero]}")

# 2 oszlop × 3 sor elrendezés (6 plot: m, v, h, F, Q, a)
fig, axs_grid = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs_grid.ravel()
fig.subplots_adjust(top=0.92, left=0.08, right=0.86, hspace=0.20, wspace=0.22, bottom=0.06)

labels = [r'$m\ \mathrm{[kg]}$', r'$v\ \mathrm{[m/s]}$', r'$h\ \mathrm{[m]}$',
          r'$F\ \mathrm{[N]}$', r'$Q\ \mathrm{[Pa]}$', r'$a/g_0\ \mathrm{[-]}$']
data = [m_vals, v_vals, h_vals, F_vals, Q_vals, a_vals]
colors = ['darkorange', 'royalblue', 'seagreen', 'crimson', 'deeppink', 'purple']

for ax, y, label, color in zip(axs, data, labels, colors):
    ax.plot(t_vals, y, color=color, lw=1.6)
    ax.set_ylabel(label, fontsize=11)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(axis='both', which='major', pad=4)

    # 0-szint fekete vízszintes vonal (m kivétel)
    if label != r'$m\ \mathrm{[kg]}$':
        ax.axhline(0.0, color='black', lw=1.0, alpha=0.85, zorder=0)

    # függőleges eseményvonalak (ha ismertek)
    if t_fuel_end is not None:
        ax.axvline(t_fuel_end, color='gray', linestyle='--', lw=1.2, zorder=3)
    if t_v_zero is not None:
        ax.axvline(t_v_zero, color='black', linestyle=':', lw=1.2, zorder=3)

# "Üzemanyag elfogy" címke az m-plot fölött, ha ismert
ax_m = axs[0]
if t_fuel_end is not None:
    y_annot = max(m_vals) if len(m_vals) else m_szaraz
    ax_m.annotate('Üzemanyag elfogy',
                  xy=(t_fuel_end, m_szaraz),
                  xycoords='data',
                  xytext=(0.3, 1.07),
                  textcoords='axes fraction',
                  ha='center', va='bottom',
                  fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray'),
                  arrowprops=dict(arrowstyle='-|>', lw=0.9, color='gray', connectionstyle='arc3,rad=0.10'),
                  zorder=6)

# 'v=0' jelölés a v-axison (ha ismert)
ax_v = axs[1]
if t_v_zero is not None:
    ax_v.annotate('$v = 0$ (tetőpont)',
                  xy=(t_v_zero, 0.0),
                  xycoords='data',
                  xytext=(0.98, 0.62),
                  textcoords='axes fraction',
                  ha='right', va='center',
                  fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='black'),
                  arrowprops=dict(arrowstyle='->', lw=0.9, color='black', connectionstyle='arc3,rad=0.0'),
                  zorder=6)

# Paraméterdoboz (jobb oldalon)
param_lines = [
    fr'$F_T = {F_t} \mathrm{{N}}$',
    fr'$\dot{{m}} = {mdot:.3g}\ \mathrm{{kg/s}}$',
    fr'$I_{{sp}} = {I_sp}\ \mathrm{{s}}$',
    fr'$k = {k:.3g}$',
    fr'$A = {A:.3g}\ \mathrm{{m^2}}$',
    fr'$m_{{0}} = {m_uzemanyag:.3g}\ \mathrm{{kg}}$',
    fr'$m_{{sz}} = {m_szaraz:.3g}\ \mathrm{{kg}}$',
    fr'$\Delta t = {dt:.3g}\ \mathrm{{s}}$'
]
param_text = '\n'.join(param_lines)

fig.text(0.875, 0.50, param_text,
         fontsize=11,
         ha='left', va='center',
         family='serif',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.95))

# Főcím és x-label
# fig.suptitle('Egyszerű rakéta szimuláció', fontsize=15)
axs[-1].set_xlabel('Idő (s)', fontsize=12)

plt.show()
