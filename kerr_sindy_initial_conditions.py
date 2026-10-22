"""SINDy identification of a squeeze-driven Kerr oscillator over many initial conditions.

    H = -Delta a^dag a + K a^dag^2 a^2 - eps2 (a^dag^2 + a^2)

Each coherent initial state is evolved exactly. SINDy identifies d<a>/dt on the training window
using a generic library of 25 candidate functions, a threshold sweep and the elbow of the Pareto
front (which includes the empty model). The identified model is integrated with the central
correlations taken from the simulation, and compared with the quantum evolution. The prediction
horizon t* is the first time the relative error exceeds tol.

Figures (saved in ./figures_initial_conditions):
    phase_space_scan.png        horizon over the phase space of Delta = 9
    energy_scan.png             horizon and correction weight versus initial energy (Delta = -10, 0, 9)
    central_region_horizon.png  horizon of 300 initial states in the central region (Delta = 9)
    central_region_error.png    relative error vs time for the SINDy model and the exact model (Delta = 9)
    classical_period.png        classical period of the centroid dynamics (Delta = 9)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"                  # one thread per worker process
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from multiprocessing import Pool
from scipy.integrate import solve_ivp

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"mathtext.fontset": "stix", "savefig.dpi": 300})

# Parameters
K, eps2 = 1.0, 3.0
CASES = {"Case I": -10.0, "Case II": 0.0, "Case III": 9.0}          # Delta of each regime
N = 140                                                             # Fock space dimension
dt, t_final, t_train, tol = 5e-4, 10.0, 2.0, 1e-2
lams = np.logspace(-12, 1, 100)                                     # threshold sweep
t = np.arange(0, t_final + dt / 2, dt)
ts = t[::2]                                                         # time grid of the integrated model
GRID = np.linspace(0.3, 6.0, 10)                                    # (q0, p0) grid, one quadrant, all regimes
cells = lambda n, h: (np.arange(n) + 0.5) * h                       # cell centers of a uniform grid
MAP_Q, MAP_P = cells(30, 0.2), cells(20, 0.2)                       # phase space of Delta = 9 (one quadrant)
ZOOM_Q, ZOOM_P = cells(10, 0.1), cells(30, 0.1)                     # central region of Delta = 9


def library(z, n, m, s):
    """Monomials in a, a* up to degree 3 and products of {1, a, a*} with the correlations."""
    zc, o = np.conj(z), np.ones_like(z)
    mono = [o, z, zc, z**2, z * zc, zc**2, z**3, z**2 * zc, z * zc**2, zc**3]
    corr = [n + 0 * z, m, np.conj(m), s, np.conj(s)]
    return np.stack(mono + [f * c for f in [o, z, zc] for c in corr], axis=-1)


def exact(D):
    """Exact coefficients of d<a>/dt in the library basis."""
    xi = np.zeros(25, complex)
    xi[[1, 2, 7, 15, 21, 13]] = [1j * D, 2j * eps2, -2j * K, -4j * K, -2j * K, -2j * K]   # a, a*, |a|^2 a, a n_c, a* m_c, s_c
    return xi


def energy(D, z):
    """Classical energy at the phase-space point alpha = z."""
    return -D * np.abs(z)**2 + K * np.abs(z)**4 - 2 * eps2 * np.real(z**2)


def critical_energies(D):
    """Separatrix energies: none (Case I), origin (Case II), hyperbolic points and origin (Case III)."""
    if D <= -2 * eps2:
        return []
    return [0.0] if D <= 2 * eps2 else [-(D - 2 * eps2)**2 / (4 * K), 0.0]


def simulate(D, alpha0):
    """Exact evolution of a coherent state; returns <a> and the central correlations n_c, m_c, s_c."""
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    ad = a.conj().T
    H = -D * ad @ a + K * ad @ ad @ a @ a - eps2 * (ad @ ad + a @ a)
    psi0 = np.ones(N, complex)
    for k in range(1, N):
        psi0[k] = psi0[k - 1] * alpha0 / np.sqrt(k)
    E, V = np.linalg.eigh(H)
    psi = V @ (np.exp(-1j * np.outer(E, t)) * (V.conj().T @ psi0 / np.linalg.norm(psi0))[:, None])
    expect = lambda O: np.sum(psi.conj() * (O @ psi), axis=0)
    al, n, m, s = expect(a), expect(ad @ a).real, expect(a @ a), expect(ad @ a @ a)
    return al, n - abs(al)**2, m - al**2, s - 2 * al * n - al.conj() * m + 2 * abs(al)**2 * al


def stlsq(Theta, y, lam):
    """Sequentially thresholded least squares."""
    xi = np.linalg.lstsq(Theta, y, rcond=None)[0]
    for _ in range(25):
        on = np.abs(xi) >= lam
        xi[~on] = 0
        if on.any():
            xi[on] = np.linalg.lstsq(Theta[:, on], y, rcond=None)[0]
    return xi


def sindy(al, n_c, m_c, s_c):
    """Threshold sweep, Pareto front with the empty model, and elbow selection."""
    y = ((al[:-4] - 8 * al[1:-3] + 8 * al[3:-1] - al[4:]) / (12 * dt))[:int(t_train / dt)]
    Theta = library(al, n_c, m_c, s_c)[2:-2][:int(t_train / dt)]
    front = {0: (np.mean(np.abs(y)**2), np.zeros(25, complex))}         # empty model
    for lam in lams:
        xi = stlsq(Theta, y, lam)
        k, mse = int((xi != 0).sum()), np.mean(np.abs(y - Theta @ xi)**2)
        if k and (k not in front or mse < front[k][0]):
            front[k] = (mse, xi)
    ks = np.array(sorted(front))
    lm = np.log10([front[k][0] for k in ks])
    x, y_ = (ks - ks[0]) / (ks[-1] - ks[0]), (lm - lm[0]) / (lm[-1] - lm[0])
    elbow = ks[np.argmax(np.abs(x[-1] * y_ - y_[-1] * x))]               # farthest point from the secant
    return front[elbow][1]


def integrate(xi, z0, n_c, m_c, s_c):
    """RK4 with step 2 dt, using the correlations on the time grid; returns <a> at ts."""
    h, f = 2 * dt, lambda z, k: library(z, n_c[k], m_c[k], s_c[k]) @ xi
    z = np.empty(len(ts), complex)
    z[0] = z0
    for j in range(len(z) - 1):
        k = 2 * j
        k1 = f(z[j], k)
        k2 = f(z[j] + h / 2 * k1, k + 1)
        k3 = f(z[j] + h / 2 * k2, k + 1)
        k4 = f(z[j] + h * k3, k + 2)
        z[j + 1] = z[j] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def run(D, q0, p0, model):
    """One initial state: identification (or exact model), integration, error and horizon."""
    alpha0 = (q0 + 1j * p0) / np.sqrt(2)
    al, n_c, m_c, s_c = simulate(D, alpha0)
    xi = sindy(al, n_c, m_c, s_c) if model == "sindy" else exact(D)
    err = np.abs(integrate(xi, al[0], n_c, m_c, s_c) - al[::2]) / np.abs(al).max()
    out = np.nonzero(err > tol)[0]
    mean_field = np.abs(1j * D * al + 2j * eps2 * al.conj() - 2j * K * abs(al)**2 * al)
    correction = np.abs(2 * K * (2 * al * n_c + al.conj() * m_c + s_c))
    return dict(D=D, q=q0, p=p0, model=model, E=energy(D, alpha0), ok=np.array_equal(xi != 0, exact(D) != 0),
                horizon=ts[out[0]] if len(out) else t_final, weight=correction.mean() / mean_field.mean(),
                err=err[::10].astype(np.float32))


def classical_period(D, z0, tmax=60):
    """Time for the classical orbit to cross its starting axis again in the same direction and side."""
    def rhs(t_, X):
        z = X[0] + 1j * X[1]
        dz = 1j * D * z + 2j * eps2 * np.conj(z) - 2j * K * abs(z)**2 * z
        return [dz.real, dz.imag]
    i, j = (1, 0) if z0.imag == 0 else (0, 1)
    X, t0 = [z0.real, z0.imag], 0.0
    side = np.sign(X[j])
    ev = lambda t_, X: X[i]
    ev.direction, ev.terminal = np.sign(rhs(0, X)[i]), True
    while t0 < tmax:
        s = solve_ivp(rhs, (t0, tmax), X, events=ev, rtol=1e-10, atol=1e-12, first_step=1e-4)
        if not s.t_events[0].size:
            return np.nan
        t0, X = s.t_events[0][0], list(s.y_events[0][0])
        if np.sign(X[j]) == side and t0 > 1e-3:
            return t0
        X = list(np.array(X) + 1e-9 * np.array(rhs(t0, X)))      # step off the axis and continue
    return np.nan


def zone_color(E, Ecs):
    """Color of an energy region: inside the separatrix, between critical energies, above them."""
    if not Ecs or E >= Ecs[-1]:
        return "darkgreen"
    return "navy" if E < Ecs[0] else "crimson"


def landscape(ax, D, qlim, plim):
    """Separatrices and well minima drawn over a phase-space map."""
    Q, P = np.meshgrid(np.linspace(-qlim, qlim, 400), np.linspace(-plim, plim, 600))
    E = energy(D, (Q + 1j * P) / np.sqrt(2))
    for Ec in critical_energies(D):
        ax.contour(Q, P, E, levels=[Ec], colors="w", linestyles="solid" if Ec < 0 else ":", linewidths=1.2, zorder=2)
    if D > -2 * eps2:                                               # two wells: mark the minima
        qw = np.sqrt((D + 2 * eps2) / K)
        ax.plot([-qw, qw], [0, 0], "D", color="royalblue", mec="w", mew=0.5, ms=4, zorder=6)
    ax.set(aspect="equal", xlim=(-qlim, qlim), ylim=(-plim, plim), xlabel=r"$q$", ylabel=r"$p$")


def horizon_map(ax, runs, qs, ps, **kw):
    """Horizon on the (q, p) grid, extended to the four quadrants by the q -> -q and p -> -p symmetry."""
    h = {(x["q"], x["p"]): x["horizon"] for x in runs}
    Q, P = np.r_[-qs[::-1], qs], np.r_[-ps[::-1], ps]
    H = np.array([[h[(qs[np.argmin(abs(qs - abs(q)))], ps[np.argmin(abs(ps - abs(p_)))])] for q in Q] for p_ in P])
    return ax.pcolormesh(Q, P, H, shading="nearest", rasterized=True, **kw)


if __name__ == "__main__":
    os.makedirs("figures_initial_conditions", exist_ok=True)
    D9 = CASES["Case III"]

    # All simulations in parallel: 300 over the three regimes, 300 + 300 in the central region of Delta = 9
    jobs = ([(D, q, p, "sindy") for D in CASES.values() for q in GRID for p in GRID]
            + [(D9, q, p, "sindy") for q in MAP_Q for p in MAP_P]
            + [(D9, q, p, m) for m in ["sindy", "exact"] for q in ZOOM_Q for p in ZOOM_P])
    with Pool(os.cpu_count() - 1) as pool:
        res = pool.starmap(run, jobs)
    n_grid, n_map = len(CASES) * len(GRID)**2, len(MAP_Q) * len(MAP_P)
    grid, phase_map = res[:n_grid], res[n_grid:n_grid + n_map]
    zoom = {m: [x for x in res[n_grid + n_map:] if x["model"] == m] for m in ["sindy", "exact"]}
    sindy_runs = [x for x in res if x["model"] == "sindy"]
    print(f"Correct SINDy models: {sum(x['ok'] for x in sindy_runs)} / {len(sindy_runs)}")

    # Prediction horizon over the phase space of Delta = 9; the dashed box is the central region
    hcmap = dict(cmap="viridis", vmin=0, vmax=t_final)
    qmax, pmax = MAP_Q[-1] + 0.1, MAP_P[-1] + 0.1
    fig, ax = plt.subplots(figsize=(3.4, 2.9), layout="constrained")
    pm = horizon_map(ax, phase_map, MAP_Q, MAP_P, **hcmap)
    landscape(ax, D9, qmax, pmax)
    yh = np.sqrt((D9 - 2 * eps2) / K)                               # hyperbolic points on the p axis
    ax.plot([0, 0], [yh, -yh], "*", color="crimson", mec="w", mew=0.4, ms=8, zorder=6)
    zq, zp = ZOOM_Q[-1] + 0.05, ZOOM_P[-1] + 0.05
    ax.plot([-zq, zq, zq, -zq, -zq], [-zp, -zp, zp, zp, -zp], "--", color="w", lw=0.8, zorder=5)
    ax.set(xticks=[-6, -3, 0, 3, 6], yticks=[-4, -2, 0, 2, 4])
    fig.colorbar(pm, ax=ax, location="bottom", shrink=0.8, aspect=35, pad=0.02, label=r"$t^*$")
    fig.savefig("figures_initial_conditions/phase_space_scan.png")

    # Horizon and correction weight versus the classical energy of the initial state
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex="col", sharey="row", layout="constrained")
    ticks = [[0, 10, 100, 1000], [-10, 0, 10, 100, 1000], [-100, -10, 0, 10, 100, 1000]]   # energy ticks per regime
    for j, D in enumerate(CASES.values()):
        r = sorted((x for x in grid if x["D"] == D), key=lambda x: x["E"])
        E = [x["E"] for x in r]
        Ecs = critical_energies(D)
        col = [zone_color(e, Ecs) for e in E]                # color by energy region, as in the error figure
        ax[0, j].scatter(E, [x["horizon"] for x in r], c=col, s=7, lw=0, zorder=3)
        ax[1, j].scatter(E, [x["weight"] for x in r], c=col, s=7, lw=0, zorder=3)
        for k in range(2):
            if len(Ecs) == 2:
                ax[k, j].axvspan(Ecs[0], Ecs[1], color="0.92", lw=0, zorder=0)   # band between critical energies
            for Ec in Ecs:
                ax[k, j].axvline(Ec, color="k", ls="-" if Ec < 0 else ":", lw=0.8)
            ax[k, j].set_xscale("symlog", linthresh=10)
            ax[k, j].set_xticks(ticks[j], [f"${v:g}$" if abs(v) <= 10 else rf"$10^{{{np.log10(v):g}}}$" if v > 0
                                           else rf"$-10^{{{np.log10(-v):g}}}$" for v in ticks[j]])
            ax[k, j].set_title(f"({'abcdef'[3 * k + j]})")
        ax[1, j].set_xlabel(r"$E$")
    ax[0, 0].set(ylabel=r"$t^*$", ylim=(0, t_final + 0.5))
    ax[1, 0].set_ylabel(r"$\overline{|\mathcal{C}|}\,/\,\overline{|\mathcal{M}|}$")
    fig.savefig("figures_initial_conditions/energy_scan.png")

    # Prediction horizon in the central region
    E_c = critical_energies(D9)[0]
    fig, ax = plt.subplots(figsize=(3.0, 5.0), layout="constrained")
    pm = horizon_map(ax, zoom["sindy"], ZOOM_Q, ZOOM_P, **hcmap)
    landscape(ax, D9, ZOOM_Q[-1] + 0.05, ZOOM_P[-1] + 0.05)
    ax.plot([0, 0], [yh, -yh], "*", color="crimson", mec="w", mew=0.4, ms=9, zorder=6)
    ax.set(xticks=[-1, 0, 1], yticks=[-3, -2, -1, 0, 1, 2, 3])
    fig.colorbar(pm, ax=ax, location="right", shrink=0.6, aspect=25, pad=0.02, label=r"$t^*$")
    fig.savefig("figures_initial_conditions/central_region_horizon.png")

    # Relative error versus time by energy zone, for the SINDy model and the exact model
    zones = [(rf"Inside separatrix ($E<{E_c:g}$)", lambda E: E < E_c, "navy"),
             (rf"Between critical energies (${E_c:g}<E<0$)", lambda E: E_c <= E < 0, "crimson"),
             (r"Above critical energy ($E>0$)", lambda E: E >= 0, "darkgreen")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, layout="constrained")
    for a, m, tag in zip(ax, ["sindy", "exact"], "ab"):
        for lab, sel, col in zones:
            err = np.array([x["err"] for x in zoom[m] if sel(x["E"])]) + 1e-16
            lo, med, hi = np.percentile(err, [25, 50, 75], axis=0)
            a.fill_between(ts[::10], lo, hi, color=col, alpha=0.15, lw=0)
            a.semilogy(ts[::10], med, color=col, lw=1.2, label=lab)
            if (med > tol).any():
                a.plot(ts[::10][np.argmax(med > tol)], tol, "o", color=col, ms=4, mec="k", mew=0.5, zorder=5)
        a.axhline(tol, color="k", lw=0.7, ls=":", label=rf"Threshold ($\delta={tol:g}$)")
        a.set(xlabel=r"$t$", title=f"({tag})", ylim=(1e-12, 1e4), xlim=(0, t_final), xticks=[0, 2, 4, 6, 8, 10],
              yticks=10.0**np.arange(-12, 1, 2))
    ax[0].set_ylabel(r"$\delta(t)=|\hat{\alpha}(t)-\alpha(t)|\,/\,\max_t|\alpha|$")
    h, l = ax[0].get_legend_handles_labels()
    ax[0].legend(h[:3], l[:3], loc="upper left", fontsize=6, frameon=False)
    ax[1].legend(h[3:], l[3:], loc="upper left", fontsize=6, frameon=False)
    fig.savefig("figures_initial_conditions/central_region_error.png")

    # Classical period of each family of orbits
    xw = np.sqrt((D9 + 2 * eps2) / (2 * K))                                          # well minimum on the Re axis
    xs = np.sqrt((D9 + 2 * eps2 - np.sqrt((D9 + 2 * eps2)**2 + 4 * K * E_c)) / (2 * K))     # separatrix on the Re axis
    ys, y0 = np.sqrt((D9 - 2 * eps2) / (2 * K)), np.sqrt((D9 - 2 * eps2) / K)        # hyperbolic point and E = 0 on the Im axis
    families = [(rf"Well orbits ($E<{E_c:g}$)", np.linspace(xs + 1e-4, xw - 1e-3, 120) + 0j, "navy", "-", 1.2),
                (rf"Central orbits (${E_c:g}<E<0$)", np.linspace(1e-3, xs - 1e-4, 60) + 0j, "crimson", "-", 2.4),
                (rf"Outer orbits (${E_c:g}<E<0$)", 1j * np.linspace(ys + 1e-4, y0 - 1e-4, 60), "darkorange", "--", 1.2),
                (r"Global orbits ($E>0$)", 1j * np.linspace(y0 + 1e-3, 3.0, 60), "darkgreen", "-", 1.2)]
    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    curves = [ax.plot(energy(D9, zs), [classical_period(D9, z) for z in zs], color=col, ls=ls, lw=lw, label=name)[0]
              for name, zs, col, ls, lw in families]
    lines = [ax.axvline(E_c, color="k", ls="-", lw=0.7, label=rf"Separatrix ($E={E_c:g}$)"),
             ax.axvline(0, color="k", ls=":", lw=0.7, label=r"Critical energy ($E=0$)"),
             ax.axhline(t_train, color="gray", ls="-.", lw=0.8, label=rf"Training window ($t={t_train:g}$)")]
    ax.set(xlabel=r"$E$", ylabel=r"$T(E)$", ylim=(0, 9))
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([-50, -10, -1, 0, 1, 10], ["$-50$", "$-10$", "$-1$", "$0$", "$1$", "$10$"])
    ax.add_artist(ax.legend(handles=curves, loc="upper right", fontsize=5.5, frameon=False))
    ax.legend(handles=lines, loc="center right", fontsize=5.5, frameon=False)
    fig.savefig("figures_initial_conditions/classical_period.png")
    plt.show()
