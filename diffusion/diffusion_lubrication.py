import numpy as np
from pathlib import Path
import scipy as sp
from numba import njit, prange
from libMobility import DPStokes, NBody
from functools import partial
import utils
from copy import deepcopy
import time

import scipy.sparse.linalg as spla
import pyamg
from tqdm import trange


def main(Lx, Ly, t_save, t_final):
    a = 1.395
    mg = 0.0592  # m*g, in pN

    L = np.array([Lx, Ly, 0], dtype=np.float64)

    kbt = 0.0041419464  # aJ
    eta = 1.4e-3 # Pa s = kg m / s

    h_g = a + (kbt / mg)
    print("gravity height: ", h_g - a)
    phi = 0.114

    tau = 6 * np.pi * eta * a**3 / kbt
    dt = 0.125
    print(f"diffusion time: {tau}, dt: {dt}")
    n_steps = int(np.ceil(t_final / dt))
    print(f"Number of steps: {n_steps}")
    assert t_save / dt % 1 == 0, 't_save was not a multiple of dt'
    n_save = int(t_save / dt)
    print(f"saving every: {n_save}")

    r_vecs = place_colloids(phi, L, a, mg, kbt)
    N = r_vecs.shape[0]
    print("z_min: ", np.min(r_vecs[:, 2]) / a)

    print("packing fraction:", N * np.pi * a**2 / (Lx * Ly))

    # sterics stuff
    firm_delta = 1e-2
    debye_length = 2.0 * a * firm_delta / np.log(10.0)
    n_cutoff = 4  # number of debye lengths to include in the cutoff
    r_cut = 2 * a + n_cutoff * debye_length
    U_0 = 4 * kbt
    nlist_buffer = 3.0  # in units of blob radius

    # solver_name = "DPStokes"
    solver_name = "NBody"
    lub_cutoff = 1e-2
    solver, lub = utils.create_solvers(
        solver_name, Lx, Ly, a, eta, lub_cut=lub_cutoff, tol=1e-1
    )
    is_periodic = solver_name != "NBody"

    output_dir = utils.get_simulation_dir(solver=solver_name, N=N, L=Lx, t_final=t_final)
    print(f"Output directory: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    F_calc = partial(
        calc_force,
        solver_name=solver_name,
        kbt=kbt,
        L=L,
        a=a,
        repulsion_strength=U_0,
        debye_length=debye_length,
        delta=firm_delta,
        mg=mg,
        N=N,
    )

    offsets, neighbor_list = utils.build_neighbor_list(
        r_vecs, r_cut + nlist_buffer * a, L, is_periodic=is_periodic
    )
    x0 = deepcopy(r_vecs)

    params = {
        "N_colloids": N,
        "Lx": Lx,
        "Ly": Ly,
        "a": a,
        "mg": mg,
        "kbt": kbt,
        "eta": eta,
        "solver_name": solver_name,
        "phi": phi,
        "diffusion_time": tau,
        "debye_length": debye_length,
        "U_0": U_0,
        "firm_delta": firm_delta,
        "lub_cutoff": lub_cutoff,
        "dt": dt,
        "T_final": t_final,
        "n_steps": n_steps,
        "r_cut": r_cut,
        "n_cutoff": n_cutoff,
        "neighbor_list_buffer": nlist_buffer,
        "t_save": t_save,
        "n_save": n_save,
    }
    utils.save_params_json(params, out_dir=output_dir)

    t_time = 20
    n_time = int(np.ceil(t_time / dt))
    t_current = 0.0
    step_start = time.time()
    rng = np.random.default_rng()

    r_vecs = r_vecs.flatten()
    for step in trange(n_steps, mininterval=20.0, desc="Simulation progress"):

        # print("zmax: ", np.max(r_vecs[2::3]) / a, "zmin: ", np.min(r_vecs[2::3]) / a)
        # print("percent overlapping wall", 100 * np.sum(r_vecs[2::3] < a) / N)

        solver.setPositions(r_vecs)

        ### brownian displacements
        delta_R = lub.delta_R_diag(r_vecs[2::3])
        sqrt_delta_R = np.sqrt(delta_R)

        W1 = rng.standard_normal(np.shape(sqrt_delta_R))
        dq = np.sqrt(2 * kbt / dt) * (
            solver.Mdot((sqrt_delta_R * W1))[0] + solver.sqrtMdotW()[0]
        )

        ### predictor velocity
        forces = F_calc(r_vecs=r_vecs, offsets=offsets, neighbor_list=neighbor_list)
        RHS = solver.Mdot(forces=forces.flatten())[0] + dq
        U_pred = solve_system(solver, lub, r_vecs, RHS, delta_R)

        ### rfd
        div_m, _ = solver.divM()

        ### predictor pos
        r_pred = r_vecs + dt * U_pred
        delta_R = lub.delta_R_diag(r_pred[2::3])

        ### corrector velocity
        solver.setPositions(r_pred)
        # NOTE: not re-computing neighbors for forces here, but OK for purely diffusive dynamics
        forces = F_calc(r_vecs=r_pred, offsets=offsets, neighbor_list=neighbor_list)
        RHS = solver.Mdot(forces=forces.flatten())[0] + (2 * kbt) * div_m + dq

        U_next = solve_system(solver, lub, r_pred, RHS, delta_R)

        r_next = r_vecs + dt * 0.5 * (U_pred + U_next)

        t_current += dt

        r_vecs = r_next

        max_delta_pos = np.max(
            np.linalg.vector_norm(r_vecs.reshape((-1, 3)) - x0, axis=1)
        )
        if max_delta_pos > nlist_buffer * a:
            temp_r = r_vecs.reshape((-1, 3))
            offsets, neighbor_list = utils.build_neighbor_list(
                temp_r, r_cut + nlist_buffer * a, L, is_periodic=is_periodic
            )
            x0 = deepcopy(temp_r)

        # if step % n_time == 0:
        #     end = time.time()
        #     elapsed = end - step_start
        #     print(f"time to simulate {n_time*dt} seconds: {elapsed:.2f}s")
        #     step_start = time.time()

        if step % n_save == 0:
            row = np.concatenate(([t_current], r_vecs.flatten()))
            utils.write_row_binary(output_dir, row, N)


def solve_system(solver, lub, r_vecs, RHS, delta_R):
    sys_size = len(r_vecs)
    RHS_norm = np.linalg.norm(RHS)

    M_diag = lub.mobility_diag(r_vecs[2::3])
    solver.setPositions(r_vecs)

    apply_A = partial(lub.apply_lubrication_matrix, solver=solver, delta_R=delta_R)
    apply_PC = partial(lub.apply_lubrication_PC, delta_R=delta_R, M_diag=M_diag)
    A_lub = spla.LinearOperator(
        (sys_size, sys_size), matvec=apply_A, dtype="float32"  # type: ignore
    )
    PC_lub = spla.LinearOperator(
        (sys_size, sys_size), matvec=apply_PC, dtype="float32"  # type: ignore
    )
    res_list = []
    (U, info) = pyamg.krylov.gmres(
        A_lub,
        (RHS / RHS_norm),
        tol=1e-1,
        M=PC_lub,
        maxiter=100,
        restrt=min(300, sys_size),
        residuals=res_list,
    )
    U *= RHS_norm

    return U


def calc_force(
    r_vecs,
    solver_name,
    kbt,
    offsets,
    neighbor_list,
    L,
    a,
    repulsion_strength,
    debye_length,
    delta,
    mg,
    N,
):

    r_vecs = np.reshape(r_vecs, (-1, 3))

    # set L to zero for non-periodic sterics
    if solver_name == "NBody":
        L_sterics = np.array([0.0, 0.0, 0.0])
    else:
        L_sterics = L

    forces = np.zeros((N, 3))
    forces += blob_blob_sterics(
        r_vectors=r_vecs,
        L=L_sterics,
        a=a,
        repulsion_strength=repulsion_strength,
        debye_length=debye_length,
        delta=delta,
        list_of_neighbors=neighbor_list,
        offsets=offsets,
    )

    if solver_name == "NBody":
        forces += blob_external_force_xy_potential_confinement_numba(
            r_vectors=r_vecs, blob_radius=a, kT=kbt, potential_width=L
        )

    forces[:, 2] += -mg

    return forces


@njit(parallel=True, fastmath=True)
def blob_external_force_xy_potential_confinement_numba(
    r_vectors, blob_radius, kT, potential_width
):
    """
    This function computes the force on a blob in a confinement potential.
    The potential has a flat bottom for 0 <= x < Lx and 0 <= y < Ly and increases quadratically outside
    """
    assert np.isfinite(kT)

    N = r_vectors.size // 3
    r_vectors = r_vectors.reshape((N, 3))
    f = np.zeros((N, 3))

    for i in prange(N):
        r = r_vectors[i, :]

        prefactor = 2 * kT / blob_radius**2
        if r[0] < 0:
            f[i, 0] += prefactor * (abs(r[0]))
        elif r[0] > potential_width[0]:
            f[i, 0] += prefactor * (potential_width[0] - r[0])

        if r[1] < 0:
            f[i, 1] += prefactor * (abs(r[1]))
        elif r[1] > potential_width[1]:
            f[i, 1] += prefactor * (potential_width[1] - r[1])

    return f


@njit(parallel=True, fastmath=True)
def blob_blob_sterics(
    r_vectors,
    L,
    a,
    repulsion_strength,
    debye_length,
    delta,
    list_of_neighbors,
    offsets,
):
    """
    The force is derived from the potential

    U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
    U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a

    with
    eps = potential strength
    r_norm = distance between blobs
    b = Debye length
    a = blob_radius
    """

    N = r_vectors.size // 3
    force = np.zeros((N, 3))

    for i in prange(N):
        # for j in range(N):
        for kk in range(offsets[i + 1] - offsets[i]):
            j = list_of_neighbors[offsets[i] + kk]

            if i == j:
                continue

            dr = np.zeros(3)
            for k in range(3):
                dr[k] = r_vectors[j, k] - r_vectors[i, k]
                if L[k] > 0:
                    dr[k] -= (
                        int(dr[k] / L[k] + 0.5 * (int(dr[k] > 0) - int(dr[k] < 0)))
                        * L[k]
                    )

            # Compute force
            r_norm = np.sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])

            offset = 2.0 * a * (1 - delta)
            temp_r = max(r_norm, 1.0e-12)
            inv_r_norm = 1 / temp_r
            if r_norm > offset:
                prefactor = (
                    -(repulsion_strength / debye_length)
                    * np.exp(-(r_norm - offset) / debye_length)
                    * inv_r_norm
                )
            else:
                prefactor = -(repulsion_strength / debye_length) * inv_r_norm

            force[i] += prefactor * dr

        # wall sterics
        h = r_vectors[i, 2]
        offset = a * (1 - delta)
        if h < offset:  # bottom wall
            force[i, 2] += repulsion_strength / debye_length
        else:
            force[i, 2] += (repulsion_strength / debye_length) * np.exp(
                -(h - offset) / debye_length
            )

    return force


def place_colloids(phi, L, a, mg, kbt):
    N = int(np.ceil(phi * L[0] * L[1] / (np.pi * a**2)))
    print(f"Number of colloids to place: {N}")

    beta = kbt / mg
    rng = np.random.default_rng()

    def sample_positions(n):
        x = rng.uniform(0, L[0], size=n)
        y = rng.uniform(0, L[1], size=n)
        z = 1.01 * a + rng.exponential(scale=beta, size=n)
        return np.stack((x, y, z), axis=1)

    pos = sample_positions(N)
    r_min = a * (2 + 0.02)

    noOverlaps = False
    it = 0
    while noOverlaps is False:
        print(f"Overlap check iteration {it+1}")
        tree = sp.spatial.cKDTree(pos, boxsize=L)
        pairs = tree.query_pairs(r=r_min)

        if not pairs:
            noOverlaps = True
            continue

        overlapping = set()
        for i, j in pairs:
            overlapping.add(i)
            overlapping.add(j)

        print(f"Found {len(pairs)} overlaps, resampling {len(overlapping)} particles")

        overlapping = sorted(overlapping)
        pos[overlapping] = sample_positions(len(overlapping))
        it += 1

    return pos


if __name__ == "__main__":
    main(
        Lx = 2560, Ly = 2560,
        t_save = 1,
        t_final = 60 * 60
    )
    main(
        Lx = 2560, Ly = 2560,
        t_save = 32,
        t_final = 8 * 60 * 60
    )
