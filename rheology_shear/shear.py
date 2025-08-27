from functools import partial
import os
import numpy as np
from libMobility import PSE
from Rigid import RigidBody
from scipy import spatial
from scipy.sparse.linalg import LinearOperator
import pyamg


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
    N_rigid = np.shape(sphere_pos)[0]
    N_blobs = N_rigid * blobs_per_body
    phi_exact = N_rigid * (4 / 3) * np.pi * rigid_radius**3 / (L[0] * L[1] * L[2])
    print(f"phi exact: {phi_exact:.4f}")

    shear = 0.0
    gamma = 1.0
    solver = PSE("periodic", "periodic", "periodic")
    split = 4 * N_blobs ** (1 / 3) / L[0]  # TODO tinker with the prefactor for speed
    solver.setParameters(shearStrain=shear, Lx=L[0], Ly=L[1], Lz=L[2], psi=split)
    solver.initialize(hydrodynamicRadius=a, viscosity=eta)

    blobs = place_blobs(sphere_pos, rigid_cfg)
    print("blob radius:", a)
    print("min dist:", np.min(spatial.distance.pdist(blobs)))

    cb = initialize_rigid_solver(rigid_cfg, sphere_pos, a, eta)
    solver.setPositions(blobs)

    def apply_A(x, n_blobs):
        out = np.zeros_like(x)
        sz = 3 * n_blobs
        lam = x[0:sz]
        U = x[sz : sz + 6 * N_rigid]

        mf = solver.Mdot(forces=lam)[0].flatten()
        kt_U = cb.K_dot(U).flatten()
        out[0:sz] = -mf + kt_U

        out[sz:] = cb.KT_dot(lam).flatten()

        return out

    N_size = 3 * N_blobs + 6 * N_rigid
    r_y = -gamma * blobs[:, 1]
    RHS = np.zeros(N_size)
    for i in range(N_blobs):
        RHS[3 * i] = r_y[i]
    RHS_norm = np.linalg.norm(RHS)
    apply_A_partial = partial(apply_A, n_blobs=N_blobs)
    A = LinearOperator((N_size, N_size), matvec=apply_A_partial, dtype="float64")  # type: ignore
    tol = 1e-4
    res_list = []
    (Sol, info_precond) = pyamg.krylov.gmres(
        A,
        (RHS / RHS_norm),
        x0=None,
        tol=tol,
        M=None,  #
        maxiter=min(300, N_size),
        restrt=None,
        residuals=res_list,
    )
    Sol *= RHS_norm
    lam = Sol[0 : 3 * N_blobs].reshape((N_blobs, 3))

    relative_r = blobs.copy()
    for i in range(N_rigid):
        relative_r[i * blobs_per_body : (i + 1) * blobs_per_body] -= sphere_pos[i]

    S = np.zeros((3, 3))
    for i in range(N_blobs):
        S += 0.5 * (np.outer(lam[i], relative_r[i]) + np.outer(relative_r[i], lam[i]))

    print(S)


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
        wall_PC=True,
        block_PC=False,
    )

    return cb


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
