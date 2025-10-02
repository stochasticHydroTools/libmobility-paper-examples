import scipy
import numpy as np
from numba import njit, prange
import json
import os
import sys

sys.path.append("./lubrication/")
from Lubrication import Lubrication


def write_row_binary(out_dir, row, N, out_dtype="float32") -> None:
    pos_file = out_dir + "colloids.bin"
    meta_file = out_dir + "binary_metadata.json"

    row = np.array(row, dtype=out_dtype)
    if not os.path.exists(meta_file):
        metadata = {
            "row_size": row.size,
            "N": N,
            "n_rows": 1,  # account for row about to be written
            "dtype": out_dtype,
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        with open(meta_file, "r") as f:
            metadata = json.load(f)
            metadata["n_rows"] += 1
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)

    with open(pos_file, "ab") as f:
        row.tofile(f)


def read_binary_file(dir) -> np.ndarray:
    pos_file = dir + "colloids.bin"
    meta_file = dir + "binary_metadata.json"

    with open(meta_file, "r") as f:
        metadata = json.load(f)

    n_rows = metadata["n_rows"]
    row_size = metadata["row_size"]
    dtype = np.dtype(metadata["dtype"])

    data = np.fromfile(pos_file, dtype=dtype)
    data = data.reshape((n_rows, row_size))
    return data


def create_solvers(
    solver_name, Lx, Ly, a, eta, lub_cut, tol=1e-1, includeAngular=False
):
    if solver_name == "NBody":
        from libMobility import NBody

        solver = NBody("open", "open", "single_wall")
        solver.setParameters(wallHeight=0.0)
        blob_fname = "./lubrication/res_coeffs/res_scalars_blob_nbody_wall_UF.csv"
    elif solver_name == "DPStokes":
        from libMobility import DPStokes

        solver = DPStokes("periodic", "periodic", "single_wall")
        solver.setParameters(
            zmin=0.0, zmax=5 * a, Lx=Lx, Ly=Ly, allowChangingBoxSize=False
        )
        blob_fname = "./lubrication/res_coeffs/res_scalars_blob_dpstokes_wall_UF.csv"
    else:
        raise ValueError(f"Unknown solver name: {solver_name}")

    mb_fname = "./lubrication/res_coeffs/res_scalars_wall_MB_2562.txt"
    lub = Lubrication(blob_fname, mb_fname, eta, a, cut=lub_cut)

    solver.initialize(viscosity=eta, hydrodynamicRadius=a, tolerance=tol)
    return solver, lub


def get_simulation_dir(solver, N, L) -> str:
    dirFound = False
    runNumber = 0
    dir = ""
    while not dirFound:
        dir = f"output/solver_{solver}_N_{N}_L_{int(L)}_run_{runNumber}/"
        if os.path.isdir(dir):
            print(f"Directory {dir} already exists, trying again...")
            runNumber += 1
            continue
        os.makedirs(dir, exist_ok=False)
        dirFound = True
    return dir


def save_params_json(params, out_dir=None):
    if out_dir is not None:
        fname = out_dir + "params.json"
    else:
        fname = "params.json"

    params["job_started_at"] = str(np.datetime64("now"))
    with open(fname, "w") as f:
        json.dump(params, f, indent=4)
    print("Saved parameters to params.json")


def build_neighbor_list(
    r_vectors,
    r_cut,
    L,
    is_periodic,
    eps=0.0,
):

    if is_periodic:
        r_vectors = periodize_r_vecs(r_vectors, L, r_vectors.shape[0])
        boxsize = L
    else:
        boxsize = None

    r_tree = scipy.spatial.cKDTree(
        r_vectors, boxsize=boxsize, balanced_tree=False, compact_nodes=False
    )

    pairs = r_tree.query_ball_point(
        r_vectors, r_cut, return_sorted=False, workers=1, eps=eps
    )  # eps has a large effect on performance and can affect accuracy if set incorrectly

    offsets = np.cumsum([0] + [len(p) for p in pairs], dtype=int)
    list_of_neighbors = np.fromiter(
        (item for sublist in pairs for item in sublist), dtype=int
    )
    return offsets, list_of_neighbors


@njit(parallel=True, fastmath=True)
def periodize_r_vecs(r_vecs_np, L, Nb):
    r_vecs = np.copy(r_vecs_np)
    # r_vecs = np.reshape(r_vecs, (Nb, 3))
    for k in prange(Nb):
        for i in range(3):
            if L[i] > 0:
                while r_vecs[k, i] < 0:
                    r_vecs[k, i] += L[i]
                while r_vecs[k, i] > L[i]:
                    r_vecs[k, i] -= L[i]
    return r_vecs
