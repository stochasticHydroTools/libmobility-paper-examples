from scipy import spatial
import numpy as np
from numba import njit, prange
import json
import os
import logging


def get_simulation_dir(mass_fact, isDeterministic, runIndex=None, loadOnly=False):
    if mass_fact == 1.0:
        dir = "hg15/"
    elif mass_fact == 0.1:
        dir = "hg61/"
    elif mass_fact == 0.2:
        dir = "hg35/"
    else:
        raise ValueError("Unknown mass factor")

    if isDeterministic:
        dir += "deterministic/"
    else:
        dir += "stochastic/"

    dirFound = False
    while not dirFound and runIndex is not None:
        dir_run = dir + f"{runIndex}/"
        if os.path.isdir(dir_run) and not loadOnly:
            logging.getLogger(__name__).warning(
                "Directory %s already exists, trying again...", dir_run
            )
            runIndex += 1
            continue

        dirFound = True
        dir = dir_run

    return dir


def save_params_json(params, out_dir=None):
    if out_dir is not None:
        fname = out_dir + "params.json"
    else:
        fname = "params.json"

    params["job_started_at"] = str(np.datetime64("now"))
    with open(fname, "w") as f:
        json.dump(params, f, indent=4)
    logging.getLogger(__name__).info("Saved parameters to %s", fname)


def build_neighbor_list(r_vectors, r_cut, eps=0.0):
    # NOTE: took periodicity out, go get it again if u want
    r_tree = spatial.cKDTree(
        r_vectors, boxsize=None, balanced_tree=False, compact_nodes=False
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
