from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

# Module-level logger
logger = logging.getLogger(__name__)

REF_DIR = Path("extracted_data")
DATA_DIR = Path("data")

N_RUNS = 3
N_C = 5
N_BOT = 2
N_TOP = 2
MARKER_SIZE = 12
LINE_WIDTH = 5
BETA_BRIGHT = 0.3  # brighten plotted series
BETA_REF = -0.4  # darken reference series
X_MIN, X_MAX = 0.0, 0.5
Y_MIN, Y_MAX = 0.5, 7.5
XTICKS = np.arange(0.0, 0.51, 0.1)
YTICKS = np.arange(1, 8, 1)
OUT_FIG = Path("comparison.png")


def brighten(rgb: np.ndarray, beta: float) -> np.ndarray:
    """
    MATLAB-like brighten:
      beta > 0 -> brighten toward white
      beta < 0 -> darken toward black
    """
    rgb = np.asarray(rgb, dtype=float)
    if beta >= 0:
        out = rgb + beta * (1.0 - rgb)
    else:
        out = rgb * (1.0 + beta)
    return np.clip(out, 0.0, 1.0)


def get_ice_colors(n_tot: int, n_bot: int, n_top: int) -> np.ndarray:
    """
    Attempt to use the 'ice' colormap from tsipkens/cmap.
    If unavailable, fall back to a perceptual matplotlib cmap.
    Slice [n_bot : n_tot - n_top] and flipud, as in MATLAB code.
    """
    # Try to import the optional 'cmap' package
    colors = None
    try:
        import cmap  # type: ignore

        ice = getattr(cmap, "ice", None)
        if callable(ice):
            colors = ice(n_tot)
    except Exception:
        colors = None

    if colors is None:
        # Fallback: use matplotlib's "viridis" (good perceptual properties)
        cm = plt.get_cmap("viridis")
        idx = np.linspace(0.0, 1.0, n_tot)
        colors = cm(idx)[:, :3]  # drop alpha if present

    colors = colors[n_bot : n_tot - n_top, :]
    colors = np.flipud(colors)
    return colors


def average_data(data_dir: Path, n_avg: int):
    """
    Python port of the MATLAB average_data function.
    Assumes CSV files: shear_stress_{i}.csv with columns [phi, series...].
    Returns:
      phi: (N,)
      avg_dat: (N, M)
      var_dat: (N, M)
    """
    # Read first to discover shape
    first = np.loadtxt(data_dir / "shear_stress_1.csv", delimiter=",")
    phi = first[:, 0]
    dat = first[:, 1:]  # N x M
    N, M = dat.shape
    all_dat = np.zeros((n_avg, N, M), dtype=dat.dtype)
    all_dat[0] = dat

    # Load remaining runs
    for i in range(2, n_avg + 1):
        fname = data_dir / f"shear_stress_{i}.csv"
        arr = np.loadtxt(fname, delimiter=",")
        if arr.shape[0] != N:
            raise ValueError(f"{fname} has {arr.shape[0]} rows, expected {N}.")
        all_dat[i - 1] = arr[:, 1:]

    avg_dat = np.mean(all_dat, axis=0)
    var_dat = np.var(all_dat, axis=0, ddof=0)
    return phi, avg_dat, var_dat


def main():
    # figure & axes
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    n_tot = N_C + N_BOT + N_TOP
    colors = get_ice_colors(n_tot, N_BOT, N_TOP)
    ref_colors = colors.copy()

    # Load averaged data
    phi, dat, var_dat = average_data(DATA_DIR, N_RUNS)
    std_err = (
        2.0 * var_dat / np.sqrt(N_RUNS)
    )  # computed but not plotted (to match MATLAB)

    # Plot the 5 main series: 1 + dat(:, i)
    handles = []
    for i in range(5):
        c = brighten(colors[i, :], BETA_BRIGHT)
        (h,) = ax.plot(
            phi,
            1.0 + dat[:, i],
            marker="s",
            markerfacecolor=c,
            markeredgecolor="none",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=c,
        )
        handles.append(h)

    # Reference curves: m = 12, 42, 162
    ref_handles = []
    for i, n_ref in enumerate([12, 42, 162]):
        fname = REF_DIR / f"{n_ref}-bead.csv"
        ref = np.loadtxt(fname, delimiter=",")
        phi_ref = ref[:, 0]
        eta_ref = ref[:, 1]
        c = brighten(ref_colors[i, :], BETA_REF)
        (h,) = ax.plot(
            phi_ref,
            eta_ref,
            linestyle="--",
            marker="D",
            markerfacecolor=c,
            markeredgecolor="none",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH - 2,
            color=c,
        )
        ref_handles.append(h)

    # Analytic curve (Ladd-like)
    phi_grid = np.linspace(0.0, 0.5, 200)
    eta_func = lambda p: (1 + 1.5 * p * (1 + p * (1 + p - 2.3 * p**2))) / (
        1 - p * (1 + p * (1 + p - 2.3 * p**2))
    )
    eta_analytic = eta_func(phi_grid)
    (h_ladd,) = ax.plot(phi_grid, eta_analytic, "-k")
    handles.extend(ref_handles)
    handles.append(h_ladd)

    # Axes formatting
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(X_MIN, X_MAX)
    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\eta_r$")

    labels = [
        "m=12",
        "m=42",
        "m=162",
        "m=642",
        "m=2562",
        "m=12 (ref.)",
        "m=42 (ref.)",
        "m=162 (ref.)",
        "Ladd (1990)",
    ]
    ax.legend(handles, labels, loc="upper left", ncols=2)

    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    fig.savefig(OUT_FIG, dpi=200)
    logger.info("Wrote %s", OUT_FIG.resolve())


if __name__ == "__main__":
    main()
