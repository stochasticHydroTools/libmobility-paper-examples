import numpy as np
import matplotlib.pyplot as plt
import utils
from pathlib import Path

"""
This is a helper script to process simulation data from the rollers simulations so that the t* distributions can be plotted using the matlab code.
"""


def main():

    a = 0.656
    Ly = 4915 * a

    ### plotting params
    nz_bins = 250
    nx_bins = 150
    nx_start = 21
    nz_start = 70

    n_runs = 4
    mass_facts = [1.0, 0.1]
    for mass_fact in mass_facts:

        get_tstar_distributions(
            n_runs, mass_fact, a, Ly, nx_bins, nz_bins, nx_start, nz_start
        )


def get_tstar_distributions(
    n_runs, mass_fact, a, Ly, nx_bins, nz_bins, nx_start, nz_start
):
    lowHeight = True if mass_fact == 1.0 else False

    t_star_arr = np.zeros(n_runs, dtype=np.float64)
    for isDeterministic in [True, False]:
        heights_dist = np.zeros(nz_bins - 1, dtype=np.float64)
        pos_dist = np.zeros(nx_bins - 1, dtype=np.float64)
        start_height_dist = np.zeros(nz_start - 1, dtype=np.float64)
        start_pos_dist = np.zeros(nx_start - 1, dtype=np.float64)
        for i in range(n_runs):
            dir = utils.get_simulation_dir(
                mass_fact, isDeterministic, runIndex=i, loadOnly=True
            )
            print(f"Processing directory: {dir}")
            fname = dir + "positions.csv"

            dat = np.loadtxt(fname, delimiter=",")

            pos = dat[:, 1:]

            k_min = 0.1 if lowHeight else 0.05
            k_max = 0.25
            percentile = 30.0
            # if lowHeight and isDeterministic:
            #     percentile = 50.0

            n_steps = pos.shape[0]
            time = dat[0:n_steps, 0]
            t_star_ind, t_star = get_t_star(
                pos, time, k_min, k_max, n_steps, percentile, plot_dir=None
            )
            t_star_arr[i] = t_star

            print(f"t_star index: {t_star_ind}, time: {time[t_star_ind]}")
            pos_t_star = pos[t_star_ind, :]
            pos_0 = pos[0, :]
            dir = utils.get_simulation_dir(mass_fact, isDeterministic)

            # Uncomment to save the positions at t* and 2t*
            # out_dir = dir + "output/"
            # np.savetxt(
            #     out_dir + f"pos_t_star_{i}.csv", pos_t_star, delimiter=",", fmt="%.6f"
            # )
            # np.savetxt(
            #     out_dir + f"pos_2t_star_{i}.csv",
            #     pos[2 * t_star_ind, :],
            #     delimiter=",",
            #     fmt="%.6f",
            # )
            # exit()

            max_height = 25 * a if lowHeight else 50 * a
            max_dist = 400 * a if lowHeight else 500 * a

            height_dist_iter, z_bins, pos_dist_iter, dist_bins = get_distrubution(
                pos_t_star, a, Ly, max_dist, max_height, nx_bins, nz_bins
            )
            (
                start_height_dist_iter,
                z_bins_start,
                start_pos_dist_iter,
                dist_bins_start,
            ) = get_distrubution(pos_0, a, Ly, max_dist, max_height, nx_start, nz_start)

            heights_dist += height_dist_iter
            pos_dist += pos_dist_iter

            start_height_dist += start_height_dist_iter
            start_pos_dist += start_pos_dist_iter

        heights_dist /= n_runs
        pos_dist /= n_runs
        start_height_dist /= n_runs
        start_pos_dist /= n_runs

        height_int = np.trapezoid(heights_dist, x=z_bins[:-1])
        pos_int = np.trapezoid(pos_dist, x=dist_bins[:-1])
        print(f"Height integral: {height_int}, Position integral: {pos_int}")

        dir = utils.get_simulation_dir(mass_fact, isDeterministic)
        out_dir = dir + "output/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving output to: {out_dir}")

        z_bins_start_mid = (z_bins_start[1:] + z_bins_start[:-1]) / 2

        output_heights_dist = np.column_stack((z_bins[1:] / a, heights_dist))
        output_pos_dist = np.column_stack((dist_bins[1:] / a, pos_dist))
        output_start_h_dist = np.column_stack((z_bins_start_mid / a, start_height_dist))
        output_start_pos_dist = np.column_stack(
            (dist_bins_start[:-1] / a, start_pos_dist)
        )

        np.savetxt(out_dir + "t_star.txt", t_star_arr, fmt="%.6f")
        np.savetxt(out_dir + "heights_dist.txt", output_heights_dist, fmt="%.6f")
        np.savetxt(out_dir + "pos_dist.txt", output_pos_dist, fmt="%.6f")
        np.savetxt(out_dir + "start_heights_dist.txt", output_start_h_dist, fmt="%.6f")
        np.savetxt(out_dir + "start_pos_dist.txt", output_start_pos_dist, fmt="%.6f")


def get_y_dist(x_pos, y_pos, Lx, Ly, delta):

    nx = int(np.round(Lx / delta))
    ny = int(np.round(Ly / delta))

    H, _, _ = np.histogram2d(x_pos, y_pos, bins=[nx, ny], range=[[0, Lx], [0, Ly]])
    H /= delta**2

    H_y = np.sum(H, axis=0)  # TODO check, sum vs mean?

    return H_y


def get_t_star(pos, time, k_min, k_max, n_steps, percentile, plot_dir=None):

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

        if plot_dir is not None and i % n_plot == 0:
            plt.figure()
            plt.plot(k_y, psd, label=f"t = {time[i]:.2f} s")
            plt.xlabel(r"$k_y$ (1/m)")
            plt.ylabel("PSD")
            plt.title(f"Power Spectral Density at t = {time[i]:.2f} s")
            plt.legend()
            plt.grid()
            plt.savefig(plot_dir + f"psd_{i:04d}.png", dpi=300)
            plt.close()

    # NOTE exclude first step because it is super peaked due to the initial condition on a grid
    print(I_vals)
    t_star_ind = np.argmax(I_vals[1:])
    t_star = time[t_star_ind]

    return t_star_ind, t_star


def get_distrubution(r_vecs, a, Ly, max_dist, max_height, nx_bins, nz_bins):

    r_vecs = np.reshape(r_vecs, (-1, 3))

    N = len(r_vecs)
    z_bins = np.linspace(0, max_height, nz_bins)
    heights = r_vecs[:, 2]
    height_dist = np.zeros(len(z_bins) - 1, dtype=np.float64)
    for i in range(len(z_bins) - 1):
        mask = (heights >= z_bins[i]) & (heights < z_bins[i + 1])
        dz = z_bins[i + 1] - z_bins[i]
        height_dist[i] = np.sum(mask) / (N * dz)

    height_dist *= a

    dist = r_vecs[:, 0]
    dist_bins = np.linspace(0, max_dist, nx_bins)
    pos_dist = np.zeros(len(dist_bins) - 1, dtype=np.float64)
    for i in range(len(dist_bins) - 1):
        mask = (dist >= dist_bins[i]) & (dist < dist_bins[i + 1])
        dx = dist_bins[i + 1] - dist_bins[i]
        pos_dist[i] = np.sum(mask) / (dx * Ly)
    pos_dist *= a

    return height_dist, z_bins, pos_dist, dist_bins


def plot_distribution(
    height_dist,
    pos_dist,
    start_heights,
    start_pos,
    z_bins,
    dist_bins,
    dist_bins_start,
    a,
    num=None,
):
    main_lw = 3
    dash_lw = 1

    f, (heights_ax, pos_ax) = plt.subplots(1, 2)
    heights_ax.plot(
        z_bins[:-1], height_dist, label="Distribution of heights", linewidth=main_lw
    )
    heights_ax.set_ylabel(r"$P(h) \times a$")
    heights_ax.set_xlabel(r"h/a")
    heights_ax.set_xlim(0, np.max(z_bins))
    height_max = max(np.max(height_dist), np.max(start_heights))
    heights_ax.set_ylim(0, 1.05 * height_max)

    heights_ax.plot(
        z_bins[:-1],
        start_heights,
        linewidth=dash_lw,
        linestyle="--",
        color="black",
    )
    pos_aspect = 50 / 0.125
    heights_ax.set_aspect(pos_aspect, adjustable="box")

    pos_ax.plot(
        dist_bins[:-1] / a,
        pos_dist,
        label="Distribution of positions",
        linewidth=main_lw,
    )
    pos_ax.set_ylabel(r"$P(x) \times a$")
    pos_ax.set_xlabel(r"x/a")
    pos_max = max(np.max(pos_dist), np.max(start_pos))
    pos_ax.set_ylim(0, pos_max * 1.05)
    pos_ax.set_xlim(0, np.max(dist_bins) / a)

    pos_aspect = 500 / 0.175
    pos_ax.set_aspect(pos_aspect, adjustable="box")

    pos_ax.plot(
        dist_bins_start[:-1] / a,
        start_pos,
        linewidth=dash_lw,
        linestyle="--",
        color="black",
    )

    # pos_ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    # plt.show()
    fname = "distributions.png"
    if num is not None:
        fname = f"distributions_{num}.png"
    plt.savefig(fname, dpi=300)


def parse_to_spunto(data, Lx, Ly, Lz, a):
    scale_factor = 10.0
    # first column is time. skip that, grab all the rest
    c_pos = data[:, 1:]
    n_steps = c_pos.shape[0]
    n_colloids = c_pos.shape[1] // 3

    c_color = 8309814
    with open(f"sim.spunto", "a") as f:
        f.write(f"#Lx={Lx};Ly={Ly};Lz={Lz};\n")

        for i in range(0, n_steps, 10):
            if i > 0:
                f.write("#\n")
            print(i)

            # f.write("# frame = " + str(i) + "\n")
            for j in range(n_colloids):
                x = c_pos[i, j * 3] * scale_factor
                y = c_pos[i, j * 3 + 1] * scale_factor
                z = c_pos[i, j * 3 + 2] * scale_factor
                r = a
                f.write(f"{x} {y} {z} {r} {c_color}\n")


if __name__ == "__main__":
    main()
