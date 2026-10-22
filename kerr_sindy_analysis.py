"""SINDy identification of the centroid dynamics of a squeeze-driven Kerr oscillator.

    H = -Delta a^dag a + K a^dag^2 a^2 - eps2 (a^dag^2 + a^2)

The centroid alpha = <a> obeys an exact equation that depends on three central correlations
of the state: n_c = <da^dag da>, m_c = <da da> and s_c = <da^dag da da>, with da = a - alpha.
SINDy recovers this equation from simulated data, and the identified model is integrated
with the correlations taken from the simulation.

Figures (saved in ./figures):
    energy_surface.png              classical energy surface of the central region
    phase_space_initial_states.png  classical energy contours and the reference initial states
    sindy_reconstruction.png        reconstruction of <q> and <p> for state A (right well)
    reference_trajectories.png      reconstruction for states B and C (near the separatrix)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"mathtext.fontset": "stix", "savefig.dpi": 300})

# Parameters
K, eps2, DELTA = 1.0, 3.0, 9.0
STATES = {"Right well": (1.8 + 0.5j, "1.8+0.5i"),              # reference initial states alpha_0
          "Center": (0.1 + 0.7j, "0.1+0.7i"),
          "Hyperbolic point": (1j * np.sqrt(1.5), r"i\sqrt{1.5}")}
N = 100                                                        # Fock space dimension
dt, t_final, t_train = 5e-4, 10.0, 2.0
lam = 0.1                                                      # STLSQ threshold
t = np.arange(0, t_final + dt / 2, dt)
ts = t[::2]                                                    # time grid of the integrated model
names = ["1", "a", "a*", "a^2", "a*^2", "|a|^2", "|a|^2 a", "a n_c", "a* m_c", "s_c"]
COLORS = {"O": "crimson", "A": "teal", "B": "forestgreen", "C": "darkviolet"}   # one color per reference state


def library(z, n, m, s):
    """Candidate functions (works for arrays and scalars)."""
    zc, r = np.conj(z), np.abs(z)**2
    return np.stack([np.ones_like(z), z, zc, z**2, zc**2, r + 0 * z, r * z, z * n, zc * m, s], axis=-1)


def energy(z):
    """Classical energy at the phase-space point alpha = z."""
    return -DELTA * np.abs(z)**2 + K * np.abs(z)**4 - 2 * eps2 * np.real(z**2)


def simulate(alpha0):
    """Exact evolution of a coherent state; returns <a> and the central correlations n_c, m_c, s_c."""
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    ad = a.conj().T
    H = -DELTA * ad @ a + K * ad @ ad @ a @ a - eps2 * (ad @ ad + a @ a)
    psi0 = np.ones(N, complex)
    for k in range(1, N):
        psi0[k] = psi0[k - 1] * alpha0 / np.sqrt(k)
    E, V = np.linalg.eigh(H)
    psi = V @ (np.exp(-1j * np.outer(E, t)) * (V.conj().T @ psi0 / np.linalg.norm(psi0))[:, None])
    expect = lambda O: np.sum(psi.conj() * (O @ psi), axis=0)
    al, n, m, s = expect(a), expect(ad @ a).real, expect(a @ a), expect(ad @ a @ a)
    return al, n - abs(al)**2, m - al**2, s - 2 * al * n - al.conj() * m + 2 * abs(al)**2 * al


def sindy(al, n_c, m_c, s_c):
    """Fourth-order finite differences + STLSQ on the training window."""
    dal = (al[:-4] - 8 * al[1:-3] + 8 * al[3:-1] - al[4:]) / (12 * dt)
    tr = slice(0, int(t_train / dt))
    Theta = library(al, n_c, m_c, s_c)[2:-2][tr]
    xi = np.linalg.lstsq(Theta, dal[tr], rcond=None)[0]
    for _ in range(25):
        on = np.abs(xi) >= lam
        xi[~on] = 0
        xi[on] = np.linalg.lstsq(Theta[:, on], dal[tr], rcond=None)[0]
    return xi


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


def tag_text(ax, *xy, tag, **kw):
    """Letter that identifies a reference state, with the same color in every figure."""
    return ax.text(*xy, tag, color=COLORS[tag], fontsize=9, fontweight="bold", zorder=12,
                   path_effects=[patheffects.withStroke(linewidth=1.5, foreground="w")], **kw)


def plot_phase_space():
    """Energy surface of the central region and phase-space map with the reference initial states."""
    E_c, E_min = -(DELTA - 2 * eps2)**2 / (4 * K), -(DELTA + 2 * eps2)**2 / (4 * K)
    points = {"O": 0j} | {tag: z0 for tag, (z0, _) in zip("ABC", STATES.values())}

    # Energy surface of the central region with its contour projection on the (q, p) plane
    fig = plt.figure(figsize=(3.4, 3.4))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    Q, P = np.meshgrid(np.linspace(-1.15, 1.15, 150), np.linspace(-2.8, 2.8, 150))
    E = energy((Q + 1j * P) / np.sqrt(2))
    floor, top = -16.0, 4.0                                         # base plane and top of the energy axis
    levels = np.linspace(E.min(), E.max(), 25)
    ax.contourf(Q, P, E, levels=levels, cmap="inferno_r", offset=floor, zorder=1)
    ax.contour(Q, P, E, levels=[E_c], colors="k", linestyles="solid", linewidths=0.8, offset=floor, zorder=2)
    ax.contour(Q, P, E, levels=[0.0], colors="k", linestyles=":", linewidths=0.8, offset=floor, zorder=2)
    surf = ax.plot_surface(Q, P, E, cmap="inferno_r", vmin=levels[0], vmax=levels[-1], rstride=6, cstride=6,
                           lw=0.2, edgecolor="k", alpha=0.95, antialiased=True, zorder=3)
    ax.contour(Q, P, E, levels=[E_c], colors="k", linestyles="solid", linewidths=1.0, zorder=4)
    ax.contour(Q, P, E, levels=[0.0], colors="k", linestyles=":", linewidths=0.9, zorder=4)
    for tag, z0 in points.items():
        q0, p0, e = np.sqrt(2) * z0.real, np.sqrt(2) * z0.imag, energy(z0)
        if abs(q0) > 1.15:                                        # outside the central window
            continue
        ax.plot([q0] * 2, [p0] * 2, [floor, e], ":", color=COLORS[tag], lw=0.7, zorder=5)
        ax.scatter([q0, q0], [p0, p0], [e, floor], s=[14, 7], color=COLORS[tag], edgecolor="w", lw=0.4, depthshade=False, zorder=6)
        dq, dp, de = {"O": (0, -0.35, 1.3), "B": (0, -0.45, 1.3), "C": (0, 0.25, 1.3)}[tag]   # label offsets
        tag_text(ax, q0 + dq, p0 + dp, e + de, tag=tag)
    ax.set(xlim=(-1.15, 1.15), ylim=(-2.8, 2.8), zlim=(floor, top), xlabel=r"$q$", ylabel=r"$p$",
           xticks=[-1, 0, 1], yticks=[-2, 0, 2], zticks=[-12, -8, -4, 0, 4])
    ax.text2D(1.02, 0.62, r"$E$", transform=ax.transAxes, fontsize=10)   # energy label drawn in front of the surface
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))                         # transparent panes
        axis._axinfo["grid"]["linewidth"] = 0
    ax.minorticks_off()
    ax.tick_params(labelsize=7, pad=-1)
    ax.xaxis.labelpad = ax.yaxis.labelpad = -3
    ax.view_init(elev=24, azim=-20)
    ax.set_box_aspect((1, 1.6, 1.1), zoom=0.9)
    fig.colorbar(surf, ax=ax, location="bottom", label=r"$E$", shrink=0.7, aspect=30, pad=0.08)
    fig.savefig("figures/energy_surface.png", bbox_inches="tight", pad_inches=0.08)

    # Phase space (q, p) with energy map, separatrix and the initial states labeled by letters
    fig, ax = plt.subplots(figsize=(3.4, 2.7), layout="constrained")
    Q, P = np.meshgrid(np.linspace(-6, 6, 600), np.linspace(-3.8, 3.8, 400))
    E = energy((Q + 1j * P) / np.sqrt(2))
    cf = ax.contourf(Q, P, E, levels=np.linspace(E_min, 20, 40), cmap="inferno_r", extend="max")
    ax.contour(Q, P, E, levels=[E_c], colors="k", linestyles="solid", linewidths=1.0)
    ax.contour(Q, P, E, levels=[0.0], colors="k", linestyles=":", linewidths=0.9)
    qw = np.sqrt((DELTA + 2 * eps2) / K)                                              # well minima on the q axis
    ax.plot([-qw, qw], [0, 0], "D", color="royalblue", mec="w", mew=0.5, ms=4, zorder=5)
    for sgn, lab in [(-1, r"$M_-$"), (1, r"$M_+$")]:                               # well minima
        ax.text(sgn * qw, -0.45, lab, color="royalblue", fontsize=9, ha="center", va="top", zorder=6,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="w")])
    offsets = {"O": (-0.75, -0.55), "A": (0.25, 0.25), "B": (0.3, -0.45), "C": (-0.75, 0.15)}
    for tag, z0 in points.items():
        q0, p0 = np.sqrt(2) * z0.real, np.sqrt(2) * z0.imag
        ax.plot(q0, p0, "o", color=COLORS[tag], mec="w", mew=0.5, ms=4, zorder=6)
        tag_text(ax, q0 + offsets[tag][0], p0 + offsets[tag][1], tag=tag)
    ax.set(aspect="equal", xlim=(-6, 6), ylim=(-3.8, 3.8), xlabel=r"$q$", ylabel=r"$p$")
    fig.colorbar(cf, ax=ax, location="bottom", label=r"$E$", ticks=[-50, -40, -30, -20, -10, 0, 10, 20],
                 shrink=0.9, aspect=35, pad=0.02)
    fig.savefig("figures/phase_space_initial_states.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_phase_space()

    # Reference states: identification, reconstruction and mean field (correlations set to zero)
    runs = {}
    for tag, (name, (alpha0, _)) in zip("ABC", STATES.items()):
        al, n_c, m_c, s_c = simulate(alpha0)
        xi = sindy(al, n_c, m_c, s_c)
        runs[tag] = (al, integrate(xi, al[0], n_c, m_c, s_c), integrate(xi, al[0], 0 * n_c, 0 * m_c, 0 * s_c))
        if tag == "A":
            print(f"Identified model ({name}): d<a>/dt =")
            for nm, c in zip(names, xi):
                if c != 0:
                    print(f"  ({c.real:+.6f} {c.imag:+.6f}i) {nm}")

    q_of = lambda z: np.sqrt(2) * z.real                       # centroid quadratures
    p_of = lambda z: np.sqrt(2) * z.imag

    # Reconstruction of state A: data in the color of the state, SINDy dashed black, training window shaded
    al, z, z_mf = runs["A"]
    col = COLORS["A"]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True, layout="constrained")
    for i, (f, lab) in enumerate([(q_of, r"$\langle q\rangle$"), (p_of, r"$\langle p\rangle$")]):
        ax[i].plot(ts, f(z_mf), color="0.65", lw=0.7, label="Mean field")
        ax[i].plot(t, f(al), color=col, lw=1.3, label="Data")
        ax[i].plot(ts, f(z), "--", color="k", lw=0.9, label="SINDy")
        ax[i].set_ylabel(lab)
    ax[2].semilogy(ts, np.abs(z - al[::2]) + 1e-16, color=col, lw=1.0)
    ax[2].set(ylabel=r"$|\hat{\alpha}-\alpha|$", xlabel=r"$t$", ylim=(1e-12, 1e-4), xlim=(0, t_final))
    ax[0].set_ylim(2.1, 6.0)
    for a_, letter in zip(ax, "abc"):
        a_.set_title(f"({letter})")
    ax[0].legend(loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.savefig("figures/sindy_reconstruction.png")

    # Reference trajectories near the separatrix: <q> (top) and error (bottom) for states B and C
    fig, ax = plt.subplots(2, 2, figsize=(5.0, 3.8), sharex=True, layout="constrained")
    for j, tag in enumerate("BC"):
        al, z, z_mf = runs[tag]
        col = COLORS[tag]
        ax[0, j].plot(ts, q_of(z_mf), color="0.65", lw=0.6)
        ax[0, j].plot(t, q_of(al), color=col, lw=1.2)
        ax[0, j].plot(ts, q_of(z), "--", color="k", lw=0.8)
        ax[1, j].semilogy(ts, np.abs(z - al[::2]) + 1e-16, color=col, lw=1.0)
        ax[1, j].set(xlabel=r"$t$", ylim=(1e-12, 1e2), xlim=(0, t_final), xticks=[0, 2, 4, 6, 8, 10],
                     yticks=10.0**np.arange(-12, 1, 4))
        for k, letter in zip(range(2), "abcd"[j::2]):
            ax[k, j].set_title(f"({letter})")
    ax[0, 0].set_ylabel(r"$\langle q\rangle$")
    ax[1, 0].set_ylabel(r"$|\hat{\alpha}-\alpha|$")
    fig.savefig("figures/reference_trajectories.png")
    plt.show()
