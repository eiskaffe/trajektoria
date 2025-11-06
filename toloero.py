import numpy as np

# ---------- top-level generator for each profile type ----------
def generate_profile(I_tot: float, t_b: float, kind: str = 'Semleges',
                     a: float = 0.8, p: float = 2.0,
                     t_r: float  = 0.116, t_drop: float = 0.10, drop_center_frac: float = 0.78,
                     tail_frac_of_mean: float = 0.05, npoints: int = 800):
    """
    Returns t (s array) and F (thrust N array).
    - I_tot: total impulse [Ns]
    - t_b: burn time [s]
    - kind: 'Regresszív' | 'Semleges' | 'Progresszív'
    - a,p: shape parameters controlling slope (see formulas)
    - t_r: rise time [s]
    - t_drop: drop (transition) duration [s]
    - drop_center_frac: center of drop in normalized time s (0..1)
    - tail_frac_of_mean: a small fractional tail level used during blending (0..1).
    """
    
    # ---------- smoothstep (quintic) ----------
    def smoothstep(u):
        """Quintic smoothstep on [0,1]."""
        u = np.clip(u, 0.0, 1.0)
        return 6*u**5 - 15*u**4 + 10*u**3

    # ---------- base g(s) definitions ----------
    def g_regressive(s, a=0.8, p=2.0):
        """Base regresszív shape: 1 - a*s^p."""
        return np.maximum(1e-12, 1.0 - a * (s**p))

    def g_neutral(s, a=0.0, p=1.0):
        """Semleges (constant)."""
        return np.ones_like(s)

    def g_progressive(s, a=0.8, p=2.0):
        """Progresszív: 1 + a*s^p."""
        return 1.0 + a * (s**p)

    # ---------- apply rise and drop smoothing (in normalized s space) ----------
    def apply_rise_and_drop(G_raw, s, t_b, t_r=0.05, t_drop=0.08, drop_center_frac=0.8, tail_level_frac=0.0):
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
                G[idx] = smoothstep(u) * G[idx]

        # Drop blending centered at drop_center_frac with half-width t_drop/2
        sd = t_drop / float(t_b) if t_b > 0 else 0.0
        if sd > 1e-12:
            start = drop_center_frac - sd/2.0
            end   = drop_center_frac + sd/2.0
            # blend in interval [start,end] from current G -> tail_level_frac
            idx_drop = (s >= start) & (s <= end)
            if np.any(idx_drop):
                u = (s[idx_drop] - start) / max(end-start, 1e-12)
                blend = smoothstep(u)
                # target tail value used here is tail_level_frac (small, e.g. 0.05)
                G[idx_drop] = (1.0 - blend) * G[idx_drop] + blend * tail_level_frac
            # after end, set to tail level
            idx_after = s > end
            if np.any(idx_after):
                G[idx_after] = tail_level_frac
        return G

    # ---------- normalize G(s) to unit integral over s in [0,1] ----------
    def normalize_G(G, s):
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
    if kind == 'Regresszív': G0 = g_regressive(s, a=a, p=p)
    elif kind == 'Progresszív': G0 = g_progressive(s, a=a, p=p)
    else: G0 = g_neutral(s)

    # t, F, I_c = generate_profileapply rise & drop smoothing (uses tail_frac_of_mean temporarily)
    # NOTE: tail_frac_of_mean is a fraction (e.g. 0.05 means tail near 5% of pre-normalized mean)
    G_blend = apply_rise_and_drop(G0, s, t_b, t_r=t_r, t_drop=t_drop,
                                  drop_center_frac=drop_center_frac, tail_level_frac=tail_frac_of_mean)

    # renormalize to ensure integral over s equals 1
    G_norm = normalize_G(G_blend, s)

    # scale by F_avg so area equals I_tot
    F_avg = float(I_tot) / float(t_b)
    F = F_avg * G_norm

    # sanity checks
    I_check = np.trapezoid(F, t)
    # small numerical rounding may occur; I_check should equal I_tot within numerical tolerance
    return t, F, I_check

# ---------- helper: analytic integrals for raw g(s) ----------
def analytic_mean_regressive(a=0.8, p=2.0):
    """Return m = integral_0^1 g_reg(s) ds = 1 - a/(p+1)"""
    return 1.0 - a/(p+1.0)

def analytic_mean_progressive(a=0.8, p=2.0):
    """m = 1 + a/(p+1)"""
    return 1.0 + a/(p+1.0)




import matplotlib.pyplot as plt
import numpy as np

def plot_three_profiles(I_tot, t_b,
                        gen_kwargs=None,
                        plot_title="Thrust profiles (Regresszív / Semleges / Progresszív)",
                        savepath=None):
    """
    Legenerálja és kirajzolja a 3 profilt ugyanazon ábrán.
    - I_tot, t_b: bemenet
    - gen_kwargs: dict az generate_profile híváshoz (a-t, p-t, t_r, t_drop, stb. átadjuk)
    - savepath: ha megadod, a kép elmentődik ide (pl. 'profiles.png')
    """
    if gen_kwargs is None:
        gen_kwargs = dict(a=0.1, p=2.2, t_r=0.716, t_drop=0.010, drop_center_frac=0.1, tail_frac_of_mean=0.025, npoints=1200)

    kinds = ['Regresszív', 'Semleges', 'Progresszív']
    labels = {'Regresszív':'Regresszív', 'Semleges':'Semleges', 'Progresszív':'Progresszív'}
    styles = {'Regresszív':'-', 'Semleges':'--', 'Progresszív':':'}
    colors = {'Regresszív':'tab:blue', 'Semleges':'tab:green', 'Progresszív':'tab:red'}

    plt.figure(figsize=(10,5))
    all_I = {}
    all_Fmax = {}
    t_common = None

    for k in kinds:
        t, F, I_check = generate_profile(I_tot, t_b, kind=k, **gen_kwargs)
        if t_common is None:
            t_common = t
        # compute F_max and its time
        idx_max = np.argmax(F)
        Fmax = float(F[idx_max])
        t_Fmax = float(t[idx_max])
        all_I[k] = I_check
        all_Fmax[k] = (Fmax, t_Fmax)

        # plot line and fill area
        plt.plot(t, F, linestyle=styles[k], label=f"{labels[k]}", linewidth=2, color=colors[k])
        # plt.fill_between(t, F, 0, alpha=0.08, color=colors[k])
    plt.axhline(0, color="black")

        # annotate max
        # plt.plot([t_Fmax], [Fmax], marker='o', color=colors[k], markersize=6)
        # plt.annotate(f"Fmax={Fmax:.0f} N\n@{t_Fmax:.2f}s", xy=(t_Fmax, Fmax),
        #              xytext=(t_Fmax+0.03*t_b, Fmax*0.9),
        #              arrowprops=dict(arrowstyle="->", lw=0.8, color=colors[k]),
        #              fontsize=9, color=colors[k])

    # draw average thrust line
    F_avg = I_tot / t_b
    plt.axhline(F_avg, color='k', linestyle=':', linewidth=1.2, label=r"$F_\text{avg}$")

    # Extras: grid, labels, legend
    plt.xlabel("Idő [s]")
    plt.ylabel("Tolóerő [N]")
    plt.title(plot_title)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best', fontsize=9)

    # show small table/text in corner with I_tot and t_b
    # textstr = f"I_tot = {I_tot:.2f} Ns\nburn time = {t_b:.2f} s"
    # plt.gca().text(0.01, 0.02, textstr, transform=plt.gca().transAxes, fontsize=9,
    #                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
        print(f"Saved figure to {savepath}")
    plt.show()

    # print summary numbers
    print("\nSummary (per profile):")
    for k in kinds:
        Fmax, tF = all_Fmax[k]
        print(f" - {labels[k]:10s}: I_check = {all_I[k]:7.2f} Ns, F_max = {Fmax:7.2f} N at t = {tF:5.3f} s")
    print(f"F_avg (I_tot/t_b) = {F_avg:.3f} N")

# --- példa használat M1315W paraméterekkel ---
I_tot = 6645.35
t_b = 5.95
gen_kwargs = dict(a=0.8, p=2, t_r=0.2, t_drop=0.7, drop_center_frac=0.96, tail_frac_of_mean=0.1, npoints=1800)

# t, F, I_c = generate_profile(I_tot, t_b)
# plt.figure(figsize=(10,5))
# print(t)
# print(F)
# print(I_c)

plot_three_profiles(I_tot, t_b, gen_kwargs, plot_title="Regresszív / Semleges / Progresszív", savepath=None)
