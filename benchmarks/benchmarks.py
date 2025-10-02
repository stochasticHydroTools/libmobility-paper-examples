import libMobility as lm
import numpy as np
import cupy as cp
import timeit
import functools
import pandas as pd


# Cache the results to avoid recomputing
@functools.lru_cache(maxsize=None)
def pack_spheres_fcc(density, lbox, min_distance, disorder=0.0):
    N = int(density * lbox**3)
    if N <= 0:
        return np.empty((0, 3), dtype=np.float64)
    a = min_distance / np.sqrt(2)
    n_cells = int(np.ceil((N / 4) ** (1 / 3)))
    i, j, k = np.mgrid[0:n_cells, 0:n_cells, 0:n_cells]
    origins = np.column_stack([i.ravel(), j.ravel(), k.ravel()]) * a
    basis = (
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
        * a
    )
    positions = np.repeat(origins, 4, axis=0) + np.tile(basis, (n_cells**3, 1))
    lattice_size = n_cells * a
    positions *= lbox / lattice_size
    if disorder > 0:
        max_displacement = disorder * min_distance
        noise = np.random.uniform(-max_displacement, max_displacement, positions.shape)
        positions += noise
        positions = positions % lbox
    return positions[:N]


def init_solver(l, use_solver="DPStokes", angular=False):
    if use_solver == "DPStokes":
        if l > 256:
            raise ValueError("Size too high for DPStokes solver. ")
        solver = lm.DPStokes(
            periodicityX="periodic", periodicityY="periodic", periodicityZ="open"
        )
        solver.setParameters(Lx=l, Ly=l, zmin=-1, zmax=l + 1)
    elif use_solver == "NBody":
        solver = lm.NBody(periodicityX="open", periodicityY="open", periodicityZ="open")
        solver.setParameters(algorithm="advise")
    elif use_solver == "NBodyWall":
        solver = lm.NBody(
            periodicityX="open", periodicityY="open", periodicityZ="single_wall"
        )
        solver.setParameters(algorithm="advise", wallHeight=-1.0)
    elif use_solver == "PSE":
        if l > 512:
            raise ValueError("Size too high for PSE solver. ")
        solver = lm.PSE(
            periodicityX="periodic", periodicityY="periodic", periodicityZ="periodic"
        )
        solver.setParameters(
            Lx=float(l), Ly=float(l), Lz=float(l), psi=0.0, shearStrain=0.0
        )
    elif use_solver == "PSE_split2_5":
        solver = lm.PSE(
            periodicityX="periodic", periodicityY="periodic", periodicityZ="periodic"
        )
        psi = max(
            1.25, 4.0 * np.sqrt(-np.log(1e-4)) / l
        )  # Ensure psi is not larger than half the box size
        print(f"Using psi: {psi:.4f} for PSE_split1_25")
        solver.setParameters(
            Lx=float(l), Ly=float(l), Lz=float(l), psi=psi, shearStrain=0.0
        )
    elif use_solver == "PSE_split1_25":
        solver = lm.PSE(
            periodicityX="periodic", periodicityY="periodic", periodicityZ="periodic"
        )
        psi = max(
            2.5, 4.0 * np.sqrt(-np.log(1e-4)) / l
        )  # Ensure psi is not larger than half the box size
        print(f"Using psi: {psi:.4f} for PSE_split2_5")
        solver.setParameters(
            Lx=float(l), Ly=float(l), Lz=float(l), psi=psi, shearStrain=0.0
        )

    elif use_solver == "DPStokes_slit20":
        solver = lm.DPStokes(
            periodicityX="periodic", periodicityY="periodic", periodicityZ="two_walls"
        )
        solver.setParameters(Lx=l, Ly=l, zmin=-10, zmax=10)
    else:
        raise ValueError(f"Unknown solver: {use_solver}")
    solver.initialize(
        viscosity=1.0,
        hydrodynamicRadius=1.0,
        includeAngular=angular,
    )
    return solver


def profile_noise(density, l, use_solver="DPStokes", angular=False):
    solver = init_solver(l, use_solver, angular=angular)

    precision = np.float32 if solver.precision == "float" else np.float64
    pos = pack_spheres_fcc(density, l, 0.5, disorder=0.25).astype(precision)
    if use_solver == "DPStokes_slit20":
        # Remove all particles that are outside the slit (z < -9 or z > 9)
        pos = pos[(pos[:, 2] > -9) & (pos[:, 2] < 9)]
    numberParticles = pos.shape[0]
    if numberParticles <= 0:
        return 0, 0.0
    if numberParticles > 2_000_000 and "NBody" in use_solver:
        raise ValueError(
            f"Number of particles ({numberParticles}) exceeds the limit for {use_solver} solver."
        )
    print(
        f"Number of particles: {numberParticles}, box size: {l:.2f}, density: {density:.3f}"
    )

    solver.setPositions(pos)
    forces = cp.ones((numberParticles, 3)).astype(precision)
    if angular:
        torques = forces.copy()
        deterministic = lambda: solver.Mdot(forces=forces, torques=torques)
    else:
        deterministic = lambda: solver.Mdot(forces=forces)

    deterministic()
    deterministic()
    nit_deterministic, time_deterministic = timeit.Timer(deterministic).autorange()
    # Time is in seconds
    solver.clean()
    cp.get_default_memory_pool().free_all_blocks()
    del solver
    return numberParticles, time_deterministic / nit_deterministic


def generate_benchmark_data(file_path: str) -> None:
    timing_file_exists = False if not pd.io.common.file_exists(file_path) else True
    timings_ds = (
        pd.read_csv(file_path)
        if timing_file_exists
        else pd.DataFrame(
            columns=[
                "solver",
                "density",
                "box_size",
                "time_deterministic",
                "gpu",
                "includes_angular",
            ]
        )
    )
    data = timings_ds.values.tolist() if not timings_ds.empty else []
    current_gpu_name = str(cp.cuda.runtime.getDeviceProperties(0)["name"])
    for angular in [False, True]:
        for s in [
            "PSE",
            "DPStokes",
            "DPStokes_slit20",
            "PSE_split1_25",
            "PSE_split2_5",
            "NBody",
            "NBodyWall",
        ]:
            print(f"Using solver: {s}")
            for d in np.logspace(-3, -1, 3):
                for l in [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
                    # Look for the current solver, density, and box size
                    is_in_ds = (
                        (timings_ds["solver"] == s)
                        & (timings_ds["density"] == d)
                        & (timings_ds["box_size"] == l)
                        & (timings_ds["includes_angular"] == angular)
                        & (timings_ds["gpu"] == current_gpu_name)
                    ).any()
                    if is_in_ds:
                        print(f"Skipping already computed: {s}, {d:.3f}, {l:.2f}")
                    else:
                        try:
                            number_particles, time_deterministic = profile_noise(
                                d, l, use_solver=s, angular=angular
                            )
                        except:
                            print(
                                f"Error profiling {s} with density {d:.3f} and box size {l:.2f}"
                            )
                            data.append(
                                [s, d, None, l, None, current_gpu_name, angular]
                            )
                        else:
                            data.append(
                                [
                                    s,
                                    d,
                                    number_particles,
                                    l,
                                    time_deterministic,
                                    current_gpu_name,
                                    angular,
                                ]
                            )
                            print(
                                f"Box size: {l:.2f}, Time deterministic: {time_deterministic:.4f} s"
                            )
                        timings = pd.DataFrame(
                            data,
                            columns=[
                                "solver",
                                "density",
                                "number_particles",
                                "box_size",
                                "time_deterministic",
                                "gpu",
                                "includes_angular",
                            ],
                        )

                        timings.to_csv(file_path, index=False)


if __name__ == "__main__":
    generate_benchmark_data("timings.csv")
