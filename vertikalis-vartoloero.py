import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import ussa1976

# Rakéta paraméterei


C_d = 0.6              # [-] A rakéta alaki tényezője / légellenelási együtthatója
D = 0.3              # [m] Rakéta átmérője
m_uzemanyag = 2      # [kg] üzemanyagtömege
m_szaraz = 4         # [kg] a rakéta 'száraztömege', üzemanyag nélkül

F_avg = 200          # [kg m s^-2] = [N] A motor névleges (átlagos) tolóereje
I_sp = 200           # [s] a rakéta hajtómű(vek) eredő specifikus impulzusa
# motor_profil = "regressive"
motor_profil = "progressive"
# motor_profil = "neutral"

D_e = 1            # [m] Az ejtőernyő átmérője
C_e = 1.6            # [-] Az ejtőernyő légellenelási együtthatója
t_infl = 1        # [s] Az ejtőernyő kinyílásának sebessége
t_kesl = 2         # [s] Az ejtőernyő kinyitásának késleltetése a tetőpont után.

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
m: float = m_uzemanyag + m_szaraz
v: float = 0.0
h: float = 0.0

# Számított paraméterek
V_e = I_sp*g_0
t = 0
v_e = V_e + 0               
A = ((D/2)**2)*np.pi                # [m^2] a rakéta homlokfelületének nagysága
A_0 = ((D_e/2)**2)*np.pi            # [m^2] az ejtőernyő homlokfelületének nagysága
# mdot = F_avg / (I_sp*g_0)         # [kg/s] tömegáram
t_0 = np.inf                        # [s] ejtőernyő kiröpítésének kezdete:
I_total = I_sp * g_0 * m_uzemanyag  # [Ns] A rakéta motor összimpulzusa
t_b = I_total / F_avg               # [s] A motor égésének ideje  

t_vals, m_vals, v_vals, h_vals, F_vals, Q_vals = [], [], [], [], [], []

def tsiolkovsky():
    return v_e * np.log((m_uzemanyag + m_szaraz) / m_szaraz)

print(f"Rakéta impulzusa")
print(f"Sanity check (Tsiolkovsky): {(voa := tsiolkovsky())}")
print(f"h_max kb @ {voa*voa / (2*g_0)}")

print("Felkészülés a szimuláció megkezdésére...", end="")

def rhoBarometrikus(h):
    return rho_0*np.e**(-h/H)

def rhoUSSA1976(z_min_m=0.0, z_max_m=30000.0, step_m=100.0):
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
rho_q_fn, _ = rhoUSSA1976(z_min_m=0.0, z_max_m=voa*voa / (2*g_0), step_m=1.0)

def generateThrustProfile(I_tot: float, t_b: float, kind: str = 'neutral',
                     a: float = 0.8, p: float = 2.0,
                     t_r: float  = 0.2, t_drop: float = 0.70, drop_center_frac: float = 0.78,
                     tail_frac_of_mean: float = 0.0, npoints: int = 1000):
    """
    Returns t (s array) and F (thrust N array).
    - I_tot: total impulse [Ns]
    - t_b: burn time [s]
    - kind: 'regressive' | 'neutral' | 'progressive'
    - a,p: shape parameters controlling slope (see formulas)
    - t_r: rise time [s]
    - t_drop: drop (transition) duration [s]
    - drop_center_frac: center of drop in normalized time s (0..1)
    - tail_frac_of_mean: a small fractional tail level used during blending (0..1).
    """
    
    # ---------- smoothStep (quintic) ----------
    def smoothStep(u):
        """Quintic smoothStep on [0,1]."""
        u = np.clip(u, 0.0, 1.0)
        return 6*u**5 - 15*u**4 + 10*u**3

    # ---------- base g(s) definitions ----------
    def g_regressive(s, a=0.8, p=2.0):
        """Base regresszív shape: 1 - a*s^p."""
        return np.maximum(1e-12, 1.0 - a * (s**p))

    def g_neutral(s, a=0.0, p=1.0):
        """Neutral (constant)."""
        return np.ones_like(s)

    def g_progressive(s, a=0.8, p=2.0):
        """Progressive: 1 + a*s^p."""
        return 1.0 + a * (s**p)

    # ---------- apply rise and drop smoothing (in normalized s space) ----------
    def applyRiseAndDrop(G_raw, s, t_b, t_r=0.0, t_drop=0.08, drop_center_frac=0.8, tail_level_frac=0.1):
        """
        G_raw: array of base g(s) values on s in [0,1].
        s: normalized time array (t/t_b)
        t_b: burn time (s)
        t_r: rise-time in seconds (0..t_b)
        t_drop: drop transition duration (seconds)
        drop_center_frac: center of drop in normalized time (0..1)
        tail_level_frac: when drop finishes, set G to this fraction of pre-normalization mean (temporary placeholder)
        """
        G = G_raw.copy()
        # Rise blending on [0, sr]
        sr = t_r / float(t_b) if t_b > 0 else 0.0
        if sr > 1e-12:
            idx = s <= sr
            if np.any(idx):
                u = s[idx] / sr  # 0..1
                G[idx] = smoothStep(u) * G[idx]

        # Drop blending centered at drop_center_frac with half-width t_drop/2
        sd = t_drop / float(t_b) if t_b > 0 else 0.0
        if sd > 1e-12:
            start = drop_center_frac - sd/2.0
            end   = drop_center_frac + sd/2.0
            # blend in interval [start,end] from current G -> tail_level_frac
            idx_drop = (s >= start) & (s <= end)
            if np.any(idx_drop):
                u = (s[idx_drop] - start) / max(end-start, 1e-12)
                blend = smoothStep(u)
                # target tail value used here is tail_level_frac (small, e.g. 0.05)
                G[idx_drop] = (1.0 - blend) * G[idx_drop] + blend * tail_level_frac
            # after end, set to tail level
            idx_after = s > end
            if np.any(idx_after):
                G[idx_after] = tail_level_frac
        return G

    # ---------- normalize G(s) to unit integral over s in [0,1] ----------
    def normalizeG(G, s):
        I = np.trapezoid(G, s)
        if I <= 0:
            raise ValueError("Integral of G non-positive.")
        return G / I
    
    if t_b <= 0:
        raise ValueError("t_b must be positive.")
    
    # time grid
    t = np.linspace(0.0, t_b, npoints)
    s = t / t_b

    # base g(s)
    if kind == 'regressive': G0 = g_regressive(s, a=a, p=p)
    elif kind == 'progressive': G0 = g_progressive(s, a=a, p=p)
    else: G0 = g_neutral(s)

    # t, F, I_c = generate_profileapply rise & drop smoothing (uses tail_frac_of_mean temporarily)
    # NOTE: tail_frac_of_mean is a fraction (e.g. 0.05 means tail near 5% of pre-normalized mean)
    G_blend = applyRiseAndDrop(G0, s, t_b, t_r=t_r, t_drop=t_drop,
                                  drop_center_frac=drop_center_frac, tail_level_frac=tail_frac_of_mean)

    # renormalize to ensure integral over s equals 1
    G_norm = normalizeG(G_blend, s)

    # scale by F_avg so area equals I_tot
    F_avg = float(I_tot) / float(t_b)
    F = F_avg * G_norm

    # sanity checks
    I_check = np.trapezoid(F, t)
    # small numerical rounding may occur; I_check should equal I_tot within numerical tolerance
    return t, F, I_check
t_F_T, F_T, I_check = generateThrustProfile(I_total, t_b, motor_profil, npoints=int(t_b//dt)+1)

def F(m, h, v, f_t):
    f = f_t # tolóerő
    f -= gamma*M_F*m / ((R_F + h)**2)   # gravitációs erő
    f -= 0.5*C_d*A*(rho_q_fn(h))*(v**2)*np.sign(v) # rakéta légellenállása
    if not np.isinf(t_0): 
        f -= 0.5*C_e*(A_0 / (1 + np.exp(-6*(t-t_0-t_kesl)/t_infl)))*(rho_q_fn(h))*(v**2)*np.sign(v) # ejtőernyő légellenállása

    return f

launched = False
i_launch: int = 0

print(f"KÉSZ!")
print("Szimuláció start.")
while t <= tmax:
    if t > 0.1 and np.average(v_vals[int(-(0.1//dt)):]) < 0 and np.isinf(t_0):
        t_0 = t
    
    f_t = F_T[int(t//dt)] if t < t_b else 0
    f = F(m, h, v, f_t)
    mdot = f_t / (I_sp*g_0)
    
    v += f*dt/m
    h += v*dt
    
    if f < 0 and not launched:
        h = 0
        v = 0
        f = 0
    elif not launched and f > 0:
        i_launch = int(np.round(t/dt))
        launched = True
        print(f"LIFTOFF! {i_launch}")
    
    if h < 0: break
    
    if m > m_szaraz: m -= mdot*dt
    
    t_vals.append(t)
    m_vals.append(m)
    v_vals.append(v)
    h_vals.append(h)
    F_vals.append(f)
    Q_vals.append(0.5*rho_q_fn(h)*(v**2))
    
    t += dt
print(f"Szimuláció vége. k={int(np.round(t/dt))}")


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

idx_v_zero = next((i for i, v in enumerate(v_vals[i_launch:]) if v <= 0), None) + i_launch
t_v_zero = t_vals[idx_v_zero] if idx_v_zero is not None else None
if idx_v_zero is not None:
    print(f"h_max = {h_vals[idx_v_zero]} m")

# 2 oszlop × 3 sor elrendezés (6 plot: m, v, h, F, Q, a)
fig, axs_grid = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs_grid.ravel()
fig.subplots_adjust(top=0.92, left=0.08, right=0.86, hspace=0.20, wspace=0.22, bottom=0.06)

labels = [r'$m\ \mathrm{[kg]}$', r'$v\ \mathrm{[m/s]}$', r'$h\ \mathrm{[m]}$',
          r'$F\ \mathrm{[N]}$', r'$Q\ \mathrm{[Pa]}$', r'$a/g_0\ \mathrm{[-]}$']
data = [m_vals, v_vals, h_vals, F_vals, Q_vals, a_vals]
colors = ['darkorange', 'royalblue', 'seagreen', 'crimson', 'deeppink', 'purple']

for ax, y, label, color in zip(axs, data, labels, colors):
    ax.plot(t_vals, y, color=color, lw=1.6, zorder=5)
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
    if not np.isinf(t_0):
        ax.axvline(t_0 + t_kesl, color='gray', linestyle='dashdot', lw=1.2, zorder=3)
        # ax.axvline(t_0 + t_kesl + t_infl, color='gray', linestyle='dashdot', lw=1.2, zorder=3)

axs[3].plot(t_F_T, F_T, linestyle=(0, (1, 1)), zorder=4, label="$F_T$")
axs[3].legend(*axs[3].get_legend_handles_labels())

# "Üzemanyag elfogy" címke az m-plot fölött, ha ismert
ax_m = axs[0]
if t_fuel_end is not None:
    y_annot = max(m_vals) if len(m_vals) else m_szaraz
    ax_m.annotate('Üzemanyag elfogy',
                  xy=(t_fuel_end, m_szaraz),
                  xycoords='data',
                  xytext=(0.98, 0.62),
                  textcoords='axes fraction',
                  ha='right', va='center',
                  fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='black'),
                  arrowprops=dict(arrowstyle='->', lw=0.9, color='black', connectionstyle='arc3,rad=0.0'),
                  zorder=6)

# 'v=0' jelölés a v-axison (ha ismert)
ax_v = axs[1]
if t_v_zero is not None:
    ax_v.annotate('$v = 0$ (tetőpont)',
                  xy=(t_v_zero, v_vals[int(t_v_zero//dt)]),
                  xycoords='data',
                  xytext=(0.98, 0.62),
                  textcoords='axes fraction',
                  ha='right', va='center',
                  fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='black'),
                  arrowprops=dict(arrowstyle='->', lw=0.9, color='black', connectionstyle='arc3,rad=0.0'),
                  zorder=6)
    
if not np.isinf(t_0):
    axs[3].annotate('ejtőernyő kinyitása',
                  xy=(t_0+t_kesl, F_vals[int((t_0+t_kesl)//dt)]),
                  xycoords='data',
                  xytext=(0.98, 0.08),
                  textcoords='axes fraction',
                  ha='right', va='center',
                  fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='black'),
                  arrowprops=dict(arrowstyle='->', lw=0.9, color='black', connectionstyle='arc3,rad=0.0'),
                  zorder=6)

# Paraméterdoboz (jobb oldalon)
param_lines = [
    fr'$F_\text{{avg}} = {F_avg}\ \mathrm{{N}}$',
    fr'$I_\text{{total}} = {I_total/1000:.3g}\ \mathrm{{kNs}}$',
    fr'$I_{{sp}} = {I_sp}\ \mathrm{{s}}$',
    fr'$C_d = {C_d:.3g}$',
    fr'$A = {A:.3g}\ \mathrm{{m^2}}$',
    fr'$m_{{0}} = {m_uzemanyag:.3g}\ \mathrm{{kg}}$',
    fr'$m_{{sz}} = {m_szaraz:.3g}\ \mathrm{{kg}}$',
    fr'$\Delta t = {dt:.3g}\ \mathrm{{s}}$',
    fr"",
    fr"$A_0 = {A_0:.3g}\ \mathrm{{m^2}}$",
    fr"$C_e = {C_e}$",
    fr"$t_\text{{késl}} = {t_kesl}\ \mathrm{{s}}$",
    fr"$t_\text{{infl}} = {t_infl}\ \mathrm{{s}}$"
]
param_text = '\n'.join(param_lines)

fig.text(0.875, 0.50, param_text,
         fontsize=11,
         ha='left', va='center',
         family='serif',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.95))

# Főcím és x-label
# fig.suptitle('Egyszerű rakéta szimuláció', fontsize=15)
axs[-2].set_xlabel('Idő (s)', fontsize=12)
axs[-1].set_xlabel('Idő (s)', fontsize=12)

print(f"Becsapódási sebesség: {v_vals[-1]} m/s")

# if (inp := input("Ábra neve > ")): plt.savefig(inp)

plt.show()
