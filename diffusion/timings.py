import numpy as np
from libMobility import NBody, DPStokes
import time
import cupy as cp
import os


def run():
    # note: may need to run fewer cases at once depending on GPU memory limitations
    phi_vals = [0.05, 0.1, 0.15, 0.25, 0.35]
    solver_names = ["dpstokes", "nbody"]
    L_vals = [100, 250, 500, 1000, 1500, 2000, 2500, 3000]
    L_vals.reverse()
    for phi in phi_vals:
        for solver_name in solver_names:
            print(f"Running timing for solver={solver_name}, phi={phi}")
            main(phi, solver_name, L_vals)


def main(phi, solver_name, L_vals):
    a = 1.0
    mg = 0.0592
    kbt = 0.0041419464
    beta = kbt / mg

    save_dir = "./output/timing_dat/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    out_file = save_dir + f"{solver_name}_phi_{phi}.txt"
    existing_data = np.loadtxt(out_file) if os.path.exists(out_file) else None
    if existing_data is not None:
        existing_Ls = existing_data[:, 0].astype(int)
    else:
        existing_Ls = []

    n_avg = 10
    new_data = []
    for L in L_vals:
        row = []

        if L in existing_Ls:
            print(f"  Found existing data for L={L}, skipping...")
            continue
        solver = create_solver(solver_name, L, a)

        N = int(phi * (L * L) / (np.pi * a**2))
        print("Running L=", "Number of particles:", N)
        pos = place_colloids(L, L, beta, phi=phi, a=a)
        solver.setPositions(pos)
        forces = cp.random.uniform(-1.0, 1.0, size=pos.shape)
        for _ in range(3):  # warm up
            solver.Mdot(forces=forces)
        for j in range(n_avg):
            pos = place_colloids(L, L, beta, phi=phi, a=a)
            solver.setPositions(pos)
            forces = cp.random.uniform(-1.0, 1.0, size=pos.shape)

            start = time.time()
            solver.Mdot(forces=forces)
            end = time.time()
            row.append(end - start)
            print("Iteration {} / {}: {:.4f} s".format(j + 1, n_avg, end - start))
        if len(row) > 0:
            row = [L] + row
            new_data.append(row)

        solver.clean()
        del solver

    with open(out_file, "a") as f:
        for row in new_data:
            f.write(" ".join(f"{val:.4f}" for val in row) + "\n")


def create_solver(solver_name, L, a):
    if solver_name.lower() == "nbody":
        solver = NBody("open", "open", "single_wall")
        solver.setParameters(wallHeight=0.0)
    elif solver_name.lower() == "dpstokes":
        solver = DPStokes("periodic", "periodic", "single_wall")
        solver.setParameters(zmin=0.0, zmax=5 * a, Lx=L, Ly=L)
    else:
        raise ValueError(f"Unknown solver name: {solver_name}")

    solver.initialize(viscosity=1.0, hydrodynamicRadius=a)
    return solver


def place_colloids(Lx, Ly, beta, phi, a):
    N = int(phi * (Lx * Ly) / (np.pi * a**2))
    L = np.array([Lx, Ly, np.inf])
    rng = np.random.default_rng()

    def sample_positions(n):
        x = rng.uniform(0, Lx, size=n)
        y = rng.uniform(0, Ly, size=n)
        z = a + rng.exponential(scale=beta, size=n)
        return np.stack((x, y, z), axis=1)

    pos = sample_positions(N)

    return pos


if __name__ == "__main__":
    run()
