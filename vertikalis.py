import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Konstansok
R_F = 6378e3        # [m] Föld sugara
M_F = 5.972e24      # [kg] Föld tömege
gamma = 6.6743e-11  # [m^3 kg^-1 s^-2] gravitációs konstans
rho_0 = 1.225       # [kg/m^3] tengerszinti levegő sűrűség
H = 8.4e3           # [m] Skálamagasság

# Paraméterek
mdot = 1            # [kg/s] tömegáram
V_e = 1000           # [m/s] exhaust velocity
k = 0.5             # [-] A rakéta alaki tényezője
A = (0.1**2)*np.pi  # [m^2] a homlokfelület nagysága
m_uzemanyag = 30   # [kg] üzemanyagtömege
m_szaraz = 6        # [kg] a rakéta 'száraztömege', üzemanyag nélkül

# Szimulációs paraméterek
dt = 0.001           # [s] lépésköz (mintavételezési) időtartam
tmax = 60     # [s] maximális vizsgált időtartam.


# Állapotváltozók
m = m_uzemanyag + m_szaraz
v = 0.0
h = 0.0

t = 0
v_e = V_e + 0

t_vals, m_vals, v_vals, h_vals, F_vals, Q_vals = [], [], [], [], [], []

def F(m, h, v):
    return mdot*v_e - gamma*M_F*m / ((R_F + h)**2) - 0.5*k*A*rho_0*(np.e**(-h/H))*(v**2)*np.sign(v)

while t <= tmax:
    f = F(m, h, v)
    
    v += f*dt/m
    h += v*dt
    
    if h < 0: break
    
    if m > m_szaraz: m -= mdot*dt
    else: v_e = 0
    
    t_vals.append(t)
    m_vals.append(m)
    v_vals.append(v)
    h_vals.append(h)
    F_vals.append(f)
    Q_vals.append(0.5*rho_0*(np.e**(-h/H))*(v**2))
    
    t += dt



# -----------------
# --- ÁBRÁZOLÁS ---
# -----------------

# Stílus: LaTeX-szerű megjelenés *LaTeX nélkül* (mathtext Computer Modern)
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern", "DejaVu Serif", "Times New Roman"],
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})

# --- Fontos események ---
# 1. Üzemanyag elfogy
idx_fuel_end = next(i for i, m in enumerate(m_vals) if m <= m_szaraz)
t_fuel_end = t_vals[idx_fuel_end]

# 2. Sebesség nullává válik (rakéta tetőpont)
idx_v_zero = next(i for i, v in enumerate(v_vals) if v <= 0)
t_v_zero = t_vals[idx_v_zero]

fig, axs = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
fig.subplots_adjust(top=0.91, left=0.09, right=0.84, hspace=0.18, bottom=0.06)  # jobb hely a paraméterdoboznak

labels = [r'$m\ \mathrm{[kg]}$', r'$v\ \mathrm{[m/s]}$', r'$h\ \mathrm{[m]}$',
          r'$F\ \mathrm{[N]}$', r'$Q\ \mathrm{[Pa]}$']
data = [m_vals, v_vals, h_vals, F_vals, Q_vals]
colors = ['darkorange', 'royalblue', 'seagreen', 'crimson', 'deeppink']

for ax, y, label, color in zip(axs, data, labels, colors):
    ax.plot(t_vals, y, color=color, lw=1.6)
    ax.set_ylabel(label, fontsize=11)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(axis='both', which='major', pad=4)

    # 0-szint fekete vízszintes vonal a v, h, F, Q plotokon
    if label != r'$m\ \mathrm{[kg]}$':
        ax.axhline(0.0, color='black', lw=1.0, alpha=0.85, zorder=0)

    # függőleges eseményvonalak (mindegyik ábrán)
    ax.axvline(t_fuel_end, color='gray', linestyle='--', lw=1.2, zorder=3)
    ax.axvline(t_v_zero, color='black', linestyle=':', lw=1.2, zorder=3)

# --- "Üzemanyag elfogy" címke csak az m-plot fölött, nem takar be ---
ax_m = axs[0]
ax_m.annotate('Üzemanyag elfogy',
              xy=(t_fuel_end, m_szaraz+2),
              xycoords='data',
              xytext=(0.3, 1.07),            # axes fraction: középen az axis felett
              textcoords='axes fraction',
              ha='center', va='bottom',
              fontsize=11,
            #   fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='gray'),
              arrowprops=dict(arrowstyle='-|>', lw=0.92, color='gray', connectionstyle='arc3,rad=0.10'),
              zorder=6)

# --- 'v=0' jelölés: doboz a v-axison jobb oldalon, nyíllal a tényleges pontra ---
ax_v = axs[1]
ax_v.annotate('$v = 0$ (tetőpont)',
              xy=(t_v_zero, 0.0),
              xycoords='data',
              xytext=(0.98, 0.62),           # axes fraction: jobb oldalon
              textcoords='axes fraction',
              ha='right', va='center',
              fontsize=11,
            #   fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='black'),
              arrowprops=dict(arrowstyle='->', lw=0.9, color='black', connectionstyle='arc3,rad=0.0'),
              zorder=6)

# --- Paraméterdoboz (jobb oldalon, szép monospace/serif formázással) ---
param_lines = [
    fr'$\dot{{m}} = {mdot:.3g}\ \mathrm{{kg/s}}$',
    fr'$v_e = {V_e}\ \mathrm{{m/s}}$',
    fr'$k = {k:.3g}$',
    fr'$A = {A:.3g}\ \mathrm{{m^2}}$',
    fr'$m_{{0}} = {m_uzemanyag:.3g}\ \mathrm{{kg}}$',
    fr'$m_{{sz}} = {m_szaraz:.3g}\ \mathrm{{kg}}$',
    fr'$\Delta t = {dt:.3g}\ \mathrm{{s}}$'
]
param_text = '\n'.join(param_lines)

fig.text(0.87, 0.50, param_text,
         fontsize=13,
         ha='left', va='center',
         family='serif',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.95))

# Főcím és x-label
fig.suptitle('Egyszerű rakéta szimuláció', fontsize=15, fontweight='semibold')
axs[-1].set_xlabel('Idő (s)', fontsize=11)

plt.show()