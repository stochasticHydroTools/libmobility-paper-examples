import matplotlib.pyplot as plt
import numpy as np
import os

self_path = os.path.dirname(os.path.abspath(__file__))

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
    dir = self_path + "/../data/"
    dat = np.loadtxt(dir + "timing_rigid_shear.csv", delimiter=",", skiprows=1)
    L_vals = dat[:, 0]
    phi_vals = dat[:, 1]
    times = dat[:, 2]
    std_times = dat[:, 3]
    iters = dat[:, 4]
    std_iters = dat[:, 5]

    colors = ["#8903c1", "#e57a1a"]
    text_colors = ["#8A14B6", "#de570fff"]
    fs = 14
    lw = 4
    ms = 12

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(
        phi_vals,
        times,
        color=colors[0],
        label="Time (s)",
        linewidth=lw,
        markersize=ms,
        marker="o",
    )
    ax2.plot(
        phi_vals,
        iters,
        color=colors[1],
        label="Iterations",
        linestyle="--",
        linewidth=lw,
        markersize=ms,
        marker="s",
    )

    ax1.set_xlabel("Volume packing fraction", fontsize=fs)
    ax1.set_ylabel("Run time (s)", color=text_colors[0], fontsize=fs)
    ax2.set_ylabel("GMRES iteration count", color=text_colors[1], fontsize=fs)
    ax1.tick_params(axis="y", labelcolor=text_colors[0], labelsize=fs)
    ax2.tick_params(axis="y", labelcolor=text_colors[1], labelsize=fs)
    ax1.ticklabel_format
    ax1.set_xlim(left=np.min(phi_vals) * 0.98, right=np.max(phi_vals))
    ax2.set_xlim(left=np.min(phi_vals) * 0.98, right=np.max(phi_vals))
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.015)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax1.set_aspect("auto")
    ax2.set_aspect("auto")

    # plt.title("Timing of Rigid Sphere Shear Simulations")
    fig.tight_layout()
    plt.savefig("timing_rigid_shear.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
