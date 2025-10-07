import os
import numpy as np
from libMobility import PSE
from Rigid import RigidBody
from scipy.sparse.linalg import LinearOperator
import pyamg
import time
import json

struct_dir = os.path.dirname(os.path.abspath(__file__)) + "/structures/"
sphere_dir = os.path.dirname(os.path.abspath(__file__)) + "/sphere_packings/configs/"


def main():

    N_vals = [12, 42, 162, 642, 2562]
    phi_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    phi_sim = np.zeros((len(N_vals), len(phi_vals)))
    s_vals = np.zeros((len(N_vals), len(phi_vals)))

    save_blob_data = False  # set to True to save blob-wise data for 3d stress plot
    start = time.time()
    for i, N in enumerate(N_vals):
        print(f"------------------- Running N ={N} -------------------")
        for j, phi in enumerate(phi_vals):
            stress, phi_exact = run(phi, N, save_blob_data)
            s_vals[i, j] = stress
            phi_sim[i, j] = phi_exact
        print(f"N={N} stress:", s_vals[i, :])

    end = time.time()
    print(f"Total simulation time: {end - start:.2f} seconds")

    save_data(phi_sim, s_vals, N_vals)


def run(phi, N, save_blob_data=False):
    struct_file = struct_dir + f"shell_N_{N}.csv"
    struct_params, rigid_cfg = load_cfg(struct_file)
    sep, blobs_per_body, rigid_rh = (
        struct_params["sep"],
        struct_params["N"],
        struct_params["Rh"],
    )

    rigid_radius_fact = 0.1
    rigid_rh *= rigid_radius_fact
    rigid_cfg *= rigid_radius_fact

    a = 0.5 * sep * rigid_radius_fact
    eta = 1.0
    L = [1.0, 1.0, 1.0]

    sphere_file = sphere_dir + f"sphere_pack_{phi:0.3g}.txt"
    sphere_pos = np.loadtxt(sphere_file, delimiter=" ", skiprows=1)
    N_rigid = np.shape(sphere_pos)[0]
    N_blobs = N_rigid * blobs_per_body
    phi_exact = N_rigid * (4 / 3) * np.pi * rigid_rh**3 / (L[0] * L[1] * L[2])
    print(f"phi exact: {phi_exact:.4f}")

    gamma = 1.0
    solver = PSE("periodic", "periodic", "periodic")
    split = 2 * N_blobs ** (1 / 3) / L[0]
    solver.setParameters(shearStrain=0.0, Lx=L[0], Ly=L[1], Lz=L[2], psi=split)
    solver.initialize(hydrodynamicRadius=a, viscosity=eta)

    blobs = place_blobs(sphere_pos, rigid_cfg)

    cb = initialize_rigid_solver(rigid_cfg, sphere_pos, a, eta)
    solver.setPositions(blobs)

    def apply_A(x):
        out = np.zeros_like(x)
        sz = 3 * N_blobs
        lam = x[0:sz]
        U = x[sz : sz + 6 * N_rigid]

        mf = solver.Mdot(forces=lam)[0].flatten()
        kt_U = cb.K_dot(U).flatten()
        out[0:sz] = -mf + kt_U

        out[sz:] = cb.KT_dot(lam).flatten()

        return out

    def apply_PC(x):
        sz = 3 * N_blobs
        out = cb.apply_PC(x[0:sz], x[sz : sz + 6 * N_rigid])
        return out

    N_size = 3 * N_blobs + 6 * N_rigid
    relative_r = blobs.copy()
    for i in range(N_rigid):
        relative_r[i * blobs_per_body : (i + 1) * blobs_per_body] -= sphere_pos[i]

    r_y = -gamma * relative_r[:, 1]
    RHS = np.zeros(N_size, dtype="float32")
    for i in range(N_blobs):
        RHS[3 * i] = r_y[i]
    RHS_norm = np.linalg.norm(RHS)
    A = LinearOperator((N_size, N_size), matvec=apply_A, dtype="float32")  # type: ignore
    PC = LinearOperator((N_size, N_size), matvec=apply_PC, dtype="float32")  # type: ignore
    tol = 1e-4
    res_list = []
    (Sol, _) = pyamg.krylov.gmres(
        A,
        (RHS / RHS_norm),
        x0=None,
        tol=tol,
        M=PC,
        maxiter=min(300, N_size),
        restrt=None,
        residuals=res_list,
    )
    print("iter count:", len(res_list))
    Sol *= RHS_norm
    lam = Sol[0 : 3 * N_blobs].reshape((N_blobs, 3))

    relative_r = blobs.copy()
    for i in range(N_rigid):
        relative_r[i * blobs_per_body : (i + 1) * blobs_per_body] -= sphere_pos[i]

    S = np.zeros((3, 3))
    stress_per_blob = np.zeros((N_blobs, 4))
    for i in range(N_blobs):
        S_i = 0.5 * (np.outer(lam[i], relative_r[i]) + np.outer(relative_r[i], lam[i]))
        S += S_i
        stress_i = S_i[0, 1] / gamma**2
        row = np.concatenate((blobs[i], [stress_i]))
        stress_per_blob[i, :] = row

    # ----- save these to make a blob-wise stress plot -------
    if save_blob_data:
        dir = "plotting/"
        with open(dir + "stress_per_blob.csv", "w") as f:
            f.write(f"# a {a}, n_per_body {blobs_per_body}\n")
            f.write("# x, y, z, stress\n")
            np.savetxt(f, stress_per_blob, delimiter=",", fmt="%0.6f")
        with open(dir + "blobs.txt", "w") as f:
            f.write(f"# a {a}, n_per_body {blobs_per_body}\n")
            np.savetxt(f, blobs, delimiter=" ")
        plot_params = {"a": a, "n_per_body": blobs_per_body}
        json.dump(plot_params, open(dir + "plot_params.json", "w"))

    print(S)
    return S[0, 1] / gamma**2, phi_exact


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


def save_data(phi_sim, s_vals, N_vals):

    save_matrix = np.zeros((phi_sim.shape[1], len(N_vals) + 1))
    save_matrix[:, 0] = phi_sim[0, :]
    save_matrix[:, 1:] = s_vals.T

    dir = "data/"
    path_found = False
    i = 1
    while not path_found:
        fname = dir + f"shear_stress_{i}.csv"
        if not os.path.exists(fname):
            path_found = True
        else:
            i += 1
    print(f"Saving to {fname}")

    np.savetxt(
        fname,
        save_matrix,
        delimiter=",",
        fmt="%0.6f",
        header="phi," + ",".join([f"N={N}" for N in N_vals]),
    )

    print("phi_sim:", phi_sim)
    print("s_vals:", s_vals)


if __name__ == "__main__":
    main()
