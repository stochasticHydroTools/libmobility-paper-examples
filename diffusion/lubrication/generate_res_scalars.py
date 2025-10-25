import re
import numpy as np
import sys
from libMobility import NBody, DPStokes

sys.path.append("..")
import utils


def main():
    eta = 1.0
    a = 1.0
    includeAngular = False
    solver_name = "NBody"

    solver = utils.create_solver(solver_name=solver_name, Lx=100 * a, Ly=100 * a, a=a, eta=eta, tol=1e-4, includeAngular=includeAngular)

    # start and end heights (in units of a)
    z_start = 1e-3
    z_end = 10
    n_heights = 10000
    heights = np.linspace(z_start, z_end, n_heights) * a
    normalize_fact = 6.0 * np.pi * eta * a

    X_tt, Y_tt = compute_res_scalars(
        heights, solver, a, normalize_fact, includeAngular=includeAngular
    )
    res_scalars = np.column_stack((heights / a, X_tt, Y_tt))

    res_scalars_fname = "res_scalars_blob_nbody_wall_trans.csv"
    dir = "./resistance_coeffs/"
    headers = ["h/a", "X_tt", "Y_tt"]
    np.savetxt(
        dir + res_scalars_fname,
        res_scalars,
        delimiter=",",
        fmt="%.8f",
        header=",".join(headers),
    )


def test_rest_scalars():
    eta = 1.523
    a = 0.954
    includeAngular = True
    ref_fname = "./resistance_coeffs/res_scalars_wall_MB.txt"
    res_scalars = np.loadtxt(ref_fname)
    res_fact = 6.0 * np.pi * eta * a

    n_heights = len(res_scalars)
    h_MB = res_scalars[:, 0]
    X_res_ref = res_scalars[:, 1]
    Y_res_ref = res_scalars[:, 2]

    solver = NBody("open", "open", "single_wall")
    solver.setParameters(wallHeight=0.0)
    solver.initialize(
        viscosity=eta, hydrodynamicRadius=a, includeAngular=includeAngular
    )

    X_tt, Y_tt = compute_res_scalars(
        h_MB, solver, a, res_fact, includeAngular=includeAngular
    )

    assert np.allclose(X_tt, X_res_ref), "X_res does not match reference"
    assert np.allclose(Y_tt, Y_res_ref), "Y_res does not match reference"
    print("All tests passed!")


# NOTE: currently only computes X_tt, Y_tt coeffs
def compute_res_scalars(heights, solver, a, normalize_fact, includeAngular=False):
    n_heights = len(heights)
    h_MB = heights

    X_tt = np.zeros(n_heights)
    Y_tt = np.zeros(n_heights)
    for i in range(n_heights):
        h_i = h_MB[i]
        r_vecs = np.array([[0.0, 0.0, a * h_i]])
        Mob = get_dense_M(solver, r_vecs, includeAngular=includeAngular)
        Res = np.linalg.inv(Mob)

        assert np.isclose(
            Res[1, 1], Res[0, 0]
        ), f"Y_tt not symmetric: {Res[1,1]} vs {Res[0,0]}"

        X_tt[i] = Res[2, 2] / normalize_fact
        Y_tt[i] = Res[0, 0] / normalize_fact

    return X_tt, Y_tt


def get_dense_M(solver, r_vecs, includeAngular=False):
    N_blobs = len(r_vecs)
    solver.setPositions(r_vecs)
    if includeAngular:
        sys_size = 6 * N_blobs
    else:
        sys_size = 3 * N_blobs

    M_dense = np.zeros((sys_size, sys_size))
    Id = np.eye(sys_size)
    for i in range(sys_size):
        forces = Id[0:3, i]
        if includeAngular:
            torques = Id[3:, i]
        else:
            torques = None
        Mf_i, Mt_i = solver.Mdot(forces=forces, torques=torques)
        if includeAngular:
            M_i = np.concatenate((Mf_i, Mt_i), axis=0)
        else:
            M_i = Mf_i
        M_dense[:, i] = M_i
    return M_dense


if __name__ == "__main__":
    main()
