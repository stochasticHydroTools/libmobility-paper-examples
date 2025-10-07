from typing import List
import numpy as np
from numba import njit, prange
from functools import partial
from copy import deepcopy
from pathlib import Path
import logging

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from libMobility import NBody

import utils


def main(mass_facts: List[float] = [0.1, 1.0], n_runs: int = 4):
    for mass_fact in tqdm(mass_facts, desc="mass factor"):
        for isDeterministic in tqdm([True, False], desc="deterministic"):
            for i in tqdm(range(n_runs), desc="runs"):
                run(mass_fact=mass_fact, isDeterministic=isDeterministic, runNumber=i)


"""
References:
[1] Balboa Usabiaga, Florencio, Blaise Delmotte, and Aleksandar Donev. 2017. “Brownian Dynamics of Confined Suspensions of Active Microrollers.” The Journal of Chemical Physics 146 (13): 134104. https://doi.org/10.1063/1.4979494.

[2] Varga, Zsigmond, Gang Wang, and James Swan. 2015. “The Hydrodynamics of Colloidal Gelation.” Soft Matter 11 (46): 9009-19. https://doi.org/10.1039/C5SM01414J.
"""


def run(mass_fact, isDeterministic, runNumber=None):
    output_dir = utils.get_simulation_dir(
        mass_fact, isDeterministic, runIndex=runNumber
    )
    logger.info("output dir: %s", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    a = 0.656  # um
    eta = 1e-3  # Pa.s

    kbt = 4.11e-3  # aJ
    g = 9.81e6  # um/s^2

    N_colloids = int(2**15)

    m = mass_fact * 1.27e-9  # milligrams
    h_g = a + (kbt / (m * g))
    logger.info("grav height h_g: %s", h_g)

    Lx = 60 * a
    Ly = 4915 * a

    freq = 10
    torque_fact = 8 * np.pi * eta * a**3 * (2 * np.pi * freq)

    r_vecs = place_colloids(N_colloids, Lx, Ly, a, m * g, kbt)
    np.savetxt(output_dir + "initial_positions.csv", r_vecs, delimiter=",", fmt="%.6f")

    debye_length = 0.1 * a
    U_0 = 4 * kbt
    tau_sterics = (6 * np.pi * eta * a**2 * debye_length) / U_0

    dt = 0.5 * tau_sterics

    out_file = output_dir + "positions.csv"

    solver = NBody("open", "open", "single_wall")
    solver.setParameters(wallHeight=0.0)
    solver.initialize(viscosity=eta, hydrodynamicRadius=a, includeAngular=True)

    # sterics stuff
    n_cutoff = 4  # number of debye lengths to include in the cutoff
    r_cut = 2 * a + n_cutoff * debye_length
    stokes_overlap_strength = (16 * np.pi * eta * a**2) / dt  # ref [2], eq. 5

    delta = 5.0  # in units of blob radius
    offsets, neighbor_list = utils.build_neighbor_list(r_vecs, r_cut + delta * a)
    x0 = deepcopy(r_vecs)

    F_calc = partial(
        calc_force_and_torque,
        L=[0, 0, 0],
        a=a,
        repulsion_strength=U_0,
        overlap_repulsion_strength=stokes_overlap_strength,
        debye_length=debye_length,
        torque_fact=torque_fact,
        mg=m * g,
        N=N_colloids,
        # r_cut=r_cut,
    )

    t_final = 0.2
    N_steps = int(np.ceil(t_final / dt))
    t_save = 0.1
    n_save = int(np.round(t_save / dt))
    logger.info("n steps: %d", N_steps)
    logger.info("saving every %d", n_save)

    params = {
        "N_colloids": N_colloids,
        "Lx": Lx,
        "Ly": Ly,
        "a": a,
        "mass_fact": mass_fact,
        "m": m,
        "kbt": kbt,
        "debye_length": debye_length,
        "U_0": U_0,
        "stokes_overlap_strength": stokes_overlap_strength,
        "tau_sterics": tau_sterics,
        "dt": dt,
        "t_final": t_final,
        "freq": freq,
        "grav_height": h_g / a,
        "delta": delta,
        "r_cut": r_cut,
        "n_cutoff": n_cutoff,
    }
    utils.save_params_json(params, output_dir)

    t_current = 0.0
    step = 0

    # initialize first step for Adams-Bashforth
    solver.setPositions(r_vecs)
    forces, torques = F_calc(r_vecs, offsets, neighbor_list)
    v_prev, _ = solver.Mdot(forces=forces, torques=torques)

    for step in tqdm(range(N_steps), desc="simulation"):
        t_current = step * dt
        if step % n_save == 0:
            with open(out_file, "a") as f:
                row = np.concatenate(([t_current], r_vecs.flatten()))
                np.savetxt(f, row.reshape(1, -1), fmt="%.8f", delimiter=",")

        forces, torques = F_calc(r_vecs, offsets, neighbor_list)
        solver.setPositions(r_vecs)
        v_det, _ = solver.Mdot(forces=forces, torques=torques)

        if not isDeterministic:
            sqrt_m, _ = solver.sqrtMdotW()
            div_m, _ = solver.divM()
            r_vecs += np.sqrt(2 * kbt * dt) * sqrt_m  # stochastic velocity
            v_det += kbt * div_m  # add drift to deterministic term

        r_vecs += dt * (1.5 * v_det - 0.5 * v_prev)
        v_prev = v_det
        max_delta_pos = np.max(np.linalg.vector_norm(r_vecs - x0, axis=1))
        if max_delta_pos > delta * a:
            offsets, neighbor_list = utils.build_neighbor_list(
                r_vecs, r_cut + delta * a
            )
            x0 = deepcopy(r_vecs)


def calc_force_and_torque(
    r_vectors,
    offsets,
    neighbor_list,
    L,
    a,
    repulsion_strength,
    overlap_repulsion_strength,
    debye_length,
    torque_fact,
    mg,
    N,
):
    forces = np.zeros((N, 3))
    forces += blob_blob_sterics(
        r_vectors=r_vectors,
        L=L,
        a=a,
        repulsion_strength=repulsion_strength,
        overlap_repulsion=overlap_repulsion_strength,
        debye_length=debye_length,
        list_of_neighbors=neighbor_list,
        offsets=offsets,
    )

    forces[:, 2] += -mg

    torques = np.zeros((N, 3))
    torques[:, 1] = torque_fact

    return forces, torques


@njit(parallel=True, fastmath=True)
def blob_blob_sterics(
    r_vectors,
    L,
    a,
    repulsion_strength,
    overlap_repulsion,
    debye_length,
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

            offset = 2.0 * a
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
        if h < a:  # bottom wall
            force[i, 2] += repulsion_strength / debye_length
            force[i, 2] += overlap_repulsion * (a / h - 1)  # ref [2], eq. 5
        else:
            force[i, 2] += (repulsion_strength / debye_length) * np.exp(
                -(h - a) / debye_length
            )

    return force


def place_colloids(N_colloids, Lx, Ly, a, mg, kbt):
    d = 2 * a
    nx = int(Lx / d)
    ny = int(Ly / d)

    beta = kbt / mg

    blob_placed = np.zeros((nx, ny), dtype=bool)
    r_vecs = np.zeros((N_colloids, 3), dtype=np.float64)

    rng = np.random.default_rng()
    n_placed = 0
    while n_placed < N_colloids:
        x = np.random.randint(0, nx)
        y = np.random.randint(0, ny)

        if not blob_placed[x, y]:
            blob_placed[x, y] = True
            r_vecs[n_placed, 0] = (x + 0.5) * d
            r_vecs[n_placed, 1] = (y + 0.5) * d

            r_vecs[n_placed, 2] = a + rng.exponential(scale=beta)

            n_placed += 1

    return r_vecs


def read_clones_file(fname):
    dat = np.loadtxt(fname, skiprows=1)
    r_vecs = dat[:, 0:3]
    return r_vecs


if __name__ == "__main__":
    main()
