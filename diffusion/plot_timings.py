import matplotlib.pyplot as plt
import numpy as np
import cmocean

plt.rcParams.update(
    {
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.family": "cmu serif",
        "font.size": 16,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


def main():

    phi_vals = [0.05, 0.1, 0.15, 0.25, 0.35]
    ymax = 12
    ymin = 1e-4
    cmap = cmocean.cm.curl
    color_step = 0.05
    color_start = 0.15
    L_actual = 2560 / 1.395
    for i, phi in enumerate(phi_vals):
        offset = color_step * i + color_start

        c1 = cmap(0.5 - offset)
        c2 = cmap(0.5 + offset)

        dir = None
        L, nbody_avg, dpstokes_avg = load_data(phi, dir=dir)

        h_nb = plt.plot(
            L, nbody_avg, label="NBody", marker="o", linestyle="-", color=c1
        )
        h_dp = plt.plot(
            L,
            dpstokes_avg,
            label="DPStokes",
            marker="s",
            linestyle="--",
            color=c2,
        )

        jump = 0.15 if i >= 2 else 0.0
        increment = 0.08 if i < 2 else 0.1
        start = 0.14
        plt.text(
            0.5 if i < 2 else 0.47,
            start + i * increment + jump,
            f"$\\phi = ${phi}",
            transform=plt.gca().transAxes,
            rotation=25 if i < 2 else 45,
            fontsize=20,
            color=c1,
        )

    h = plt.vlines(L_actual, ymin, ymax, colors="black", linestyles="--")
    plt.ylim(bottom=ymin, top=ymax)
    plt.rcParams["legend.framealpha"] = 1.0  # solid frame
    plt.rcParams["legend.edgecolor"] = "black"  # black border
    plt.rcParams["legend.fancybox"] = False  # square corners
    plt.legend(
        [h_nb[0], h_dp[0], h],
        ["NBody", "DPStokes", "$L/a$ from Section 5.2"],
        loc="upper left",
        fontsize=12,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(
        [100, 250, 500, 1000, 2000, 3000], ["100", "250", "500", "1000", "2000", "3000"]
    )
    plt.ylim(top=ymax)
    ax = plt.gca()
    ax.set_aspect(2900 / 10**4)
    plt.xlabel("Dimensionless box size, $L/a$")
    plt.ylabel("Runtime (s)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig("diffusion_timings.svg", dpi=300)
    plt.close()


def load_data(phi, dir=None):
    if dir is None:
        data_dir = "./timing_dat/"
    else:
        data_dir = "./timing_dat/" + dir + "/"
    fname_nbody = data_dir + f"nbody_phi_{phi}.txt"
    fname_dpstokes = data_dir + f"dpstokes_phi_{phi}.txt"
    dat_nbody = np.loadtxt(fname_nbody)
    dat_dpstokes = np.loadtxt(fname_dpstokes)
    L_nbody = dat_nbody[:, 0]
    L_dpstokes = dat_dpstokes[:, 0]
    nbody_times_mean = np.mean(dat_nbody[:, 1:3], axis=1)
    # nbody_times_std = np.std(dat_nbody[:, 1:3], axis=1)
    dpstokes_times_mean = np.mean(dat_dpstokes[:, 1:3], axis=1)
    # dpstokes_times_std = np.std(dat_dpstokes[:, 1:3], axis=1)

    nbody_times_mean = nbody_times_mean[np.argsort(L_nbody)]
    dpstokes_times_mean = dpstokes_times_mean[np.argsort(L_dpstokes)]

    L = np.sort(L_nbody)

    return L, nbody_times_mean, dpstokes_times_mean


def plot_timings(phi, ind, ymax):
    plt.subplot(2, 2, ind)
    ax = plt.gca()

    L_vals, nbody_avg, dpstokes_avg = load_data(phi)

    ax.plot(
        L_vals,
        nbody_avg,
        label="NBody",
        marker="o",
        linestyle="--",
    )
    ax.plot(
        L_vals,
        dpstokes_avg,
        label="DPStokes",
        marker="s",
        linestyle="--",
    )
    if ind in [1, 3]:
        ax.set_ylabel("Time (s)")
        ax.tick_params(axis="y", which="both")
    if ind in [3, 4]:
        ax.set_xlabel("Box size L")
    if ind in [2, 4]:
        ax.tick_params(axis="y", which="both", labelleft=False)
    if ind in [1, 2]:
        ax.tick_params(axis="x", which="both", labelbottom=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=ymax)

    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", length=4)
    ax.tick_params(axis="both", which="minor", length=2)

    plt.grid(True, which="both", ls="--", lw=0.5)
    ax.text(0.7, 0.1, f"$\\phi = ${phi}", transform=ax.transAxes, fontsize=15)
    if ind == 1:
        ax.legend()


if __name__ == "__main__":
    main()
