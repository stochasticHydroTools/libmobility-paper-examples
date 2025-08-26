import os
import numpy as np
from libMobility import PSE
from Rigid import RigidBody
import scipy
from scipy import spatial


def main():
    struct_dir = "structures/"
    struct_file = struct_dir + "shell_N_162.csv"
    struct_params, rigid_cfg = load_cfg(struct_file)
    sep, blobs_per_body, rigid_radius = (
        struct_params["sep"],
        struct_params["N"],
        struct_params["Rh"],
    )

    # this number is the goal radius from the sphere generation
    # it is slightly smaller than the final radius so there are no overlaps
    rigid_radius_fact = 0.1
    rigid_radius *= rigid_radius_fact
    rigid_cfg *= rigid_radius_fact

    a = 0.5 * sep * rigid_radius_fact
    eta = 1.0
    L = [1.0, 1.0, 1.0]

    phi = 0.3

    sphere_file = f"sphere_packings/configs/sphere_pack_{phi:0.3g}.txt"
    sphere_pos = np.loadtxt(sphere_file, delimiter=" ", skiprows=1)
    N = np.shape(sphere_pos)[0]
    N_blobs = N * blobs_per_body
    phi_exact = N * (4 / 3) * np.pi * rigid_radius**3 / (L[0] * L[1] * L[2])
    print(f"phi exact: {phi_exact:.4f}")

    shear = 1.0
    solver = PSE("periodic", "periodic", "periodic")
    split = 4 * N_blobs ** (1 / 3) / L[0]  # TODO tinker with the prefactor for speed
    solver.setParameters(shearStrain=shear, Lx=L[0], Ly=L[1], Lz=L[2], psi=split)
    solver.initialize(hydrodynamicRadius=a, viscosity=eta)

    blobs = place_blobs(sphere_pos, rigid_cfg)


def place_blobs(sphere_pos, rigid_cfg):
    N = sphere_pos.shape[0]
    N_per_body = rigid_cfg.shape[0]

    blobs = np.zeros((N * N_per_body, 3))

    for i in range(N):
        sphere_blobs = rigid_cfg.copy()
        center_i = sphere_pos[i]
        sphere_blobs += center_i
        blobs[i * N_per_body : (i + 1) * N_per_body] = sphere_blobs

    return blobs


def load_cfg(file_name):
    with open(file_name, "r") as f:
        _ = f.readline()
        params = f.readline().strip().split(",")
        sep = float(params[0].split(" ")[1])
        N = int(params[1])
        rg = float(params[2])
        rh = int(params[3])
        cfg = np.loadtxt(f, delimiter=" ")
        params = {"sep": sep, "N": N, "Rg": rg, "Rh": rh}
    return params, cfg


if __name__ == "__main__":
    main()
