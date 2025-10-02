import time
import numpy as np
from scipy import interpolate


class Lubrication:

    def __init__(self, res_scalars_blob_fname, res_scalars_mb_fname, eta, a, cut=1e-3):
        blob_res_scalars = np.loadtxt(
            res_scalars_blob_fname, comments="#", delimiter=","
        )
        h_blob = blob_res_scalars[:, 0]
        x_blob = blob_res_scalars[:, 1]
        y_blob = blob_res_scalars[:, 2]

        self.X_R_blob_interp = interpolate.interp1d(
            h_blob, x_blob, kind="cubic", fill_value="extrapolate"  # type: ignore
        )
        self.Y_R_blob_interp = interpolate.interp1d(
            h_blob, y_blob, kind="cubic", fill_value="extrapolate"  # type: ignore
        )

        self.res_fact = 6.0 * np.pi * eta * a
        self.a = a

        ref_MB_res_scalars = np.loadtxt(res_scalars_mb_fname)

        h_MB = ref_MB_res_scalars[:, 0]
        X_MB = ref_MB_res_scalars[:, 1]
        Y_MB = ref_MB_res_scalars[:, 2]
        self.X_r_ref_interp = interpolate.interp1d(
            h_MB, X_MB, kind="cubic", fill_value="extrapolate"  # type: ignore
        )
        self.Y_r_ref_interp = interpolate.interp1d(
            h_MB, Y_MB, kind="cubic", fill_value="extrapolate"  # type: ignore
        )

        self.x_cutoff = 0.3
        self.y_cutoff = 0.15
        self.cut = cut  # minimum height to avoid blow up

    def delta_R_diag(self, heights):
        """
        Compute the diagonal of the Delta_R matrix for lubrication corrections.

        Parameters:
        heights : np.ndarray
            vector of particle heights
        Returns:
        np.ndarray
            The diagonal of the Delta_R matrix.
        """
        N = len(heights)
        DR = np.zeros(3 * N)
        h_nd = heights / self.a  # non-dimensional height

        # Set a minimum h value to avoid massive blow-up in Brenner approximation
        h_nd_cut = np.clip(h_nd, 1.0 + self.cut, None)
        epsilon = h_nd_cut - 1.0

        exact_res_X = np.zeros_like(h_nd)
        exact_res_Y = np.zeros_like(h_nd)

        # this uses the Brenner approximation for small epsilon
        # and the 2562 multiblob approximation for large epsilon
        eps_x_inds_brenner = np.where(epsilon < self.x_cutoff)[0]
        eps_y_inds_brenner = np.where(epsilon < self.y_cutoff)[0]
        eps_x_inds_MB = np.where(epsilon >= self.x_cutoff)[0]
        eps_y_inds_MB = np.where(epsilon >= self.y_cutoff)[0]

        if len(eps_x_inds_brenner) > 0:
            exact_res_X[eps_x_inds_brenner] = (
                1.0 / epsilon[eps_x_inds_brenner]
                - (1.0 / 5.0) * np.log(epsilon[eps_x_inds_brenner])
                + 0.971280
            )
        if len(eps_y_inds_brenner) > 0:
            exact_res_Y[eps_y_inds_brenner] = (
                -(8.0 / 15.0) * np.log(epsilon[eps_y_inds_brenner]) + 0.9588
            )

        if len(eps_x_inds_MB) > 0:
            exact_res_X[eps_x_inds_MB] = self.X_r_ref_interp(h_nd[eps_x_inds_MB])
        if len(eps_y_inds_MB) > 0:
            exact_res_Y[eps_y_inds_MB] = self.Y_r_ref_interp(h_nd[eps_y_inds_MB])

        X_res_blob = self.X_R_blob_interp(h_nd)
        Y_res_blob = self.Y_R_blob_interp(h_nd)
        DR[0::3] = exact_res_Y - Y_res_blob
        DR[1::3] = exact_res_Y - Y_res_blob
        DR[2::3] = exact_res_X - X_res_blob
        return self.res_fact * DR

    def sqrt_delta_R(self, heights):
        """
        Compute the square root of the diagonal of the Delta_R matrix.
        Negative eigenvalues are set to zero.

        Parameters:
        heights : np.ndarray
            vector of particle heights
        Returns:
        np.ndarray
            The square root of the diagonal of the Delta_R matrix.
        """
        DR = self.delta_R_diag(heights)
        assert np.all(DR >= 0), "Negative values in Delta_R diagonal"
        return np.sqrt(DR)

    def apply_lubrication_matrix(self, x, solver, delta_R):
        # r_vecs = r_vecs.flatten()
        # want to apply (I + M*Delta_R)x
        # DR = self.delta_R_diag(r_vecs[2::3])
        DR_x = delta_R * x

        M_DR_x, _ = solver.Mdot(DR_x)
        return x + M_DR_x

    def apply_lubrication_PC(self, x, delta_R, M_diag):
        # r_vecs = r_vecs.flatten()
        # heights = r_vecs[2::3]
        # M_diag = self.mobility_diag(heights)
        # DR = self.delta_R_diag(heights)
        lub_inv = 1.0 / (1.0 + M_diag * delta_R)
        return lub_inv * x

    def mobility_diag(self, h_vec):
        """
        Compute the diagonal of the mobility matrix.

        Parameters:
        X_mb_eval : interpolation object
            Interpolate X^tt_MB from data.
        Y_mb_eval : interpolation object
            Interpolate Y^tt_MB from data.
        h_vec : np.ndarray
            The height vector
        a : float
            The radius of the particles.
        res_fact : float
            The resistance factor.

        Returns:
        np.ndarray
            The diagonal of the mobility matrix.
        """
        N = len(h_vec)
        Mob_diag = np.zeros(3 * N)
        h_nd = h_vec / self.a  # non-dimensional height
        h_nd = np.clip(
            h_nd, 1.0 + self.cut, None
        )  # Set a minimum value to avoid division by zero

        Mob_diag[0::3] = 1.0 / (self.Y_R_blob_interp(h_nd) * self.res_fact)
        Mob_diag[1::3] = 1.0 / (self.Y_R_blob_interp(h_nd) * self.res_fact)
        Mob_diag[2::3] = 1.0 / (self.X_R_blob_interp(h_nd) * self.res_fact)
        return Mob_diag
