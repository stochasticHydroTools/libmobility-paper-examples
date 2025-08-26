import os
import numpy as np
import scipy
from scipy import spatial


def main():

    L = [1.0, 1.0, 1.0]
    Rh = 0.1

    phi_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    for phi in phi_vals:
        fname = f"configs/sphere_pack_{phi:0.3g}.txt"
        if not os.path.exists(fname):
            continue

        print(f"----------------checking phi {phi}----------------")

        sphere_pos = np.loadtxt(fname, delimiter=" ", skiprows=1)
        N = np.shape(sphere_pos)[0]
        phi_exact = N * (4 / 3) * np.pi * Rh**3 / (L[0] * L[1] * L[2])
        assert np.isclose(phi, phi_exact, rtol=1e-1)
        print(f"phi calc: {phi_exact:.4f}")

        assert np.max(sphere_pos) <= L[0]
        assert np.min(sphere_pos) >= 0.0

        r_tree = spatial.cKDTree(
            sphere_pos, boxsize=L, balanced_tree=False, compact_nodes=False
        )

        for r_vec in sphere_pos:
            ind = r_tree.query_ball_point(r_vec, 2.0 * Rh)
            if len(ind) > 1:
                print(f"Sphere at {r_vec} is overlapping with spheres: {ind}")

        min_dist = spatial.distance.pdist(sphere_pos).min()
        print(min_dist)
        assert min_dist >= 2.0 * Rh

        print("PASSED")


def read_spheres(filename, num_spheres=-1):
    """
    Read spheres from a binary file into an (N, 4) NumPy array.
    Each row is (x, y, z, diameter), stored as little-endian float64.
    """
    data = np.fromfile(filename, dtype="<f8")  # little-endian float64
    return data.reshape((num_spheres, 4))


def load_cfg(file_name):
    with open(file_name, "r") as f:
        _ = f.readline()
        params = f.readline().strip().split(",")
        sep = float(params[0].split(" ")[1])
        N = int(params[1])
        rg = float(params[2])
        rh = int(params[3])
        cfg = np.loadtxt(f, delimiter=" ", skiprows=2)
        params = {"sep": sep, "N": N, "Rg": rg, "Rh": rh}
    return params, cfg


if __name__ == "__main__":
    main()
