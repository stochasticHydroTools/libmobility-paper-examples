import numpy as np
from libMobility import PSE
from scipy.sparse.linalg import LinearOperator
import pyamg
import os
import utils
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/sphere_packings")
import generate_timing_configs as tc

struct_dir = os.path.dirname(os.path.abspath(__file__)) + "/structures/"
sphere_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/sphere_packings/timing_configs/"
)


def main():
    sphere_N = 642
    struct_file = struct_dir + f"shell_N_{sphere_N}.csv"
    struct_params, rigid_cfg = utils.load_cfg(struct_file)
    sep, blobs_per_body, rigid_rh = (
        struct_params["sep"],
        struct_params["N"],
        struct_params["Rh"],
    )
    rigid_radius_fact = 0.05
    rigid_rh *= rigid_radius_fact
    rigid_cfg *= rigid_radius_fact

    a = 0.5 * sep * rigid_radius_fact
    N_spheres = 1000

    L_vals = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0])
    phi_vals = N_spheres * (4 / 3) * np.pi * rigid_rh**3 / (L_vals**3)
    n_avg = 3
    times = np.zeros((n_avg, len(L_vals)))
    iters = np.zeros((n_avg, len(L_vals)))
    for j in range(n_avg):
        tc.generate_configs()
        for i, L in enumerate(L_vals):
            time_taken, n_iters = run(L, N_spheres, rigid_cfg, a)
            print(f"L={L}, time: {time_taken:.4f}s, iters: {n_iters}")
            times[j, i] = time_taken
            iters[j, i] = n_iters

    avg_times = np.mean(times, axis=0)
    avg_iters = np.mean(iters, axis=0)
    std_times = np.std(times, axis=0)
    std_iters = np.std(iters, axis=0)

    save_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/"
    out_file = save_dir + "timing_rigid_shear.csv"
    dat = np.zeros((len(L_vals), 6))
    dat[:, 0] = L_vals
    dat[:, 1] = phi_vals
    dat[:, 2] = avg_times
    dat[:, 3] = std_times
    dat[:, 4] = avg_iters
    dat[:, 5] = std_iters
    np.savetxt(
        out_file,
        dat,
        delimiter=",",
        header="L, phi, time(s), time_std(s), iters, iters_std",
        comments="",
        fmt="%.4f",
    )
    print(f"Saved timing data to {out_file}")


def run(L, N_rigid, rigid_cfg, a):
    eta = 1.0
    gamma = 1.0

    sphere_file = sphere_dir + f"sphere_pack_{L}.txt"
    sphere_pos = np.loadtxt(sphere_file, delimiter=" ", skiprows=1)
    N_rigid = np.shape(sphere_pos)[0]
    blobs_per_body = np.shape(rigid_cfg)[0]
    N_blobs = N_rigid * blobs_per_body

    gamma = 1.0
    solver = PSE("periodic", "periodic", "periodic")
    split = 2 * N_blobs ** (1 / 3) / L
    solver.setParameters(shearStrain=0.0, Lx=L, Ly=L, Lz=L, psi=split)
    solver.initialize(hydrodynamicRadius=a, viscosity=eta)

    blobs = utils.place_blobs(sphere_pos, rigid_cfg)

    cb = utils.initialize_rigid_solver(rigid_cfg, sphere_pos, a, eta)
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
    start = time.time()
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
    end = time.time()
    return end - start, len(res_list)


if __name__ == "__main__":
    main()
