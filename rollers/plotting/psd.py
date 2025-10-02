import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

"""
Computes t_star by computing PSD and creates a plot of the PSD integral over time. Useful for finding k_min cutoffs since you can test to see where the PSD integral starts to plateau/decay.
"""


def main():

    dir = "../hg15/deterministic/"
    # dir = "../hg15/stochastic/"
    # dir = "../hg61/stochastic/"
    # dir = "../hg61/deterministic/"
    print(f"Using directory: {dir}")

    plot_dir = dir + "plots/"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    k_min = 0.1
    k_max = 0.25
    cutoff_percentile = 30.0

    n_runs = 1
    for i in range(n_runs):
        fname = dir + f"{i}/positions.csv"
        dat = np.loadtxt(fname, delimiter=",")
        pos = dat[:, 1:]
        time = dat[:, 0]

        n_steps = pos.shape[0]
        time = time[:n_steps]
        plot_fname = f"psd_integral_{i}.png"
        get_t_star(
            pos, time, k_min, k_max, n_steps, cutoff_percentile, plot_dir + plot_fname
        )


def get_y_dist(x_pos, y_pos, Lx, Ly, delta):

    nx = int(np.round(Lx / delta))
    ny = int(np.round(Ly / delta))

    H, _, _ = np.histogram2d(x_pos, y_pos, bins=[nx, ny], range=[[0, Lx], [0, Ly]])
    H /= delta**2

    H_y = np.sum(H, axis=0)

    return H_y


def get_t_star(pos, time, k_min, k_max, n_steps, percentile, plot_file=None):

    I_vals = np.zeros(n_steps, dtype=np.float64)

    for i in range(n_steps):
        pos_t = pos[i, :]

        x_cutoff = np.percentile(pos_t[0::3], percentile)
        front_indices = (pos_t[0::3] > x_cutoff).nonzero()[0]
        Lx = np.max(pos_t[0::3])
        Ly = np.max(pos_t[1::3])

        x_front = pos_t[0::3][front_indices]
        y_front = pos_t[1::3][front_indices]

        ny = 400
        delta = Ly / ny
        dist = get_y_dist(x_pos=x_front, y_pos=y_front, Lx=Lx, Ly=Ly, delta=delta)

        density_fluctuations = dist - np.mean(dist)

        fft_vals = np.fft.fft(density_fluctuations)

        k_y = np.fft.fftfreq(ny, d=delta) * 2 * np.pi

        psd = np.abs(fft_vals) ** 2 / ny

        k_y = np.fft.fftshift(k_y)
        psd = np.fft.fftshift(psd)

        mask = (k_y >= k_min) & (k_y <= k_max)
        k_y = k_y[mask]
        psd = psd[mask]

        I_vals[i] = np.trapezoid(psd, x=k_y)

    # NOTE exclude first step because it is super peaked due to the initial condition on a grid
    t_star_ind = np.argmax(I_vals[1:])
    t_star = time[t_star_ind]

    if plot_file is not None:
        plt.figure()
        plt.plot(time, I_vals, label="Integral of PSD")
        plt.xlabel("Time (s)")
        plt.ylabel("Integral of PSD")
        plt.text(
            0.05,
            0.95,
            f"$k_{{min}}$ = {k_min:.3f}\n$k_{{max}}$ = {k_max:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        plt.axvline(
            x=t_star, color="red", linestyle="--", label=f"$t^* = {t_star:.2f}$"
        )
        plt.text(
            t_star,
            np.max(I_vals) * 0.95,
            f"$t^* = {t_star:.2f}$",
            color="red",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        plt.title("Integral of Power Spectral Density over Time")
        plt.grid()
        print(f"Saving plot to {plot_file}")
        plt.savefig(plot_file, dpi=300)

    return t_star_ind, t_star


if __name__ == "__main__":
    main()
