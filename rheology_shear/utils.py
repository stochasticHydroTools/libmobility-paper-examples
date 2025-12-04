import numpy as np
from Rigid import RigidBody


def initialize_rigid_solver(rigid_cfg, sphere_pos, a, eta):
    N = sphere_pos.shape[0]
    Q = np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]]), N, axis=0)
    cb = RigidBody(
        rigid_cfg,
        X=sphere_pos,
        Q=Q,
        a=a,
        eta=eta,
        dt=0.0,
        wall_PC=False,
        block_PC=False,
    )

    return cb


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
