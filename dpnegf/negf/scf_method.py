import numpy as np
import logging

log = logging.getLogger(__name__)
class PDIISMixer:
    """
    Periodic Direct Inversion in the Iterative Subspace (PDIIS) mixer for accelerating SCF convergence.

    Parameters
    ----------
    init_p : np.ndarray
        Initial potential or state vector for SCF iterations.
    mix_rate : float, optional
        Mixing rate (step size) for linear update. Default is 0.05.
    n_history : int, optional
        Number of history steps to store for Pulay extrapolation. Default is 6.
    mixing_period : int, optional
        Frequency (in iterations) to apply DIIS mixing instead of linear mixing. Default is 3.
    verbose : bool, optional
        If True, print debug information. Default is False.
    """
    def __init__(self, init_p, mix_rate=0.05, n_history=4, mixing_period=2, verbose=True):
        assert isinstance(init_p, np.ndarray), "init_p must be a numpy array"
        
        self.mix_rate = mix_rate
        self.n_history = n_history
        self.mixing_period = mixing_period
        self.verbose = verbose
        
        self.iter_count = 0
        self.p = init_p.copy()
        self.f = None
        self.R = [None for _ in range(n_history)]
        self.F = [None for _ in range(n_history)]

    def reset(self, new_init_p=None):
        """Reset the mixer, optionally with a new initial potential."""
        self.iter_count = 0
        self.f = None
        self.R = [None for _ in range(self.n_history)]
        self.F = [None for _ in range(self.n_history)]
        if new_init_p is not None:
            assert isinstance(new_init_p, np.ndarray), "new_init_p must be a numpy array"
            self.p = new_init_p.copy()

    def update(self, p_new):
        """
        Perform one PDIIS mixing update based on the new input p_new.

        Parameters
        ----------
        p_new : np.ndarray
            Newly computed state (e.g., electrostatic potential).

        Returns
        -------
        p_next : np.ndarray
            The next mixed state.
        """
        assert isinstance(p_new, np.ndarray), "p_new must be a numpy array"
        assert p_new.shape == self.p.shape, "Shape mismatch in p_new and current state"
        
        p_new = p_new.copy()
        f_new = p_new - self.p

        if self.f is not None:
            idx = self.iter_count % self.n_history
            self.R[idx] = p_new - self.p # Residual vector
            self.F[idx] = f_new - self.f # Difference in residuals

        

        do_pdiis = (self.iter_count + 1) % self.mixing_period == 0
        p_next = None

        if do_pdiis and all(f is not None for f in self.F):
            if self.verbose:
                log.info(msg=f"[PDIIS] Performing DIIS mixing at iter {self.iter_count + 1}")
            F_mat = np.stack(self.F, axis=1)
            R_mat = np.stack(self.R, axis=1)

            FtF = F_mat.T @ F_mat

            try:
                cond_FtF = np.linalg.cond(FtF)
                if cond_FtF > 1e10:
                    log.info(f"[PDIIS DEBUG] cond(FtF) = {cond_FtF:.2e}")
                    log.info(f"[PDIIS DEBUG] Norms of F vectors: {[np.linalg.norm(f) for f in self.F]}")
                    log.info(f"[PDIIS DEBUG] Rank of F_mat: {np.linalg.matrix_rank(F_mat)}")
                    log.info(msg=f"[PDIIS] Warning: FtF matrix condition number too high ({cond_FtF:.2e}). Skipping DIIS.")
                    raise RuntimeError("Ill-conditioned FtF matrix in PDIIS")

                correction = (R_mat + self.mix_rate * F_mat) @ np.linalg.solve(FtF, F_mat.T @ f_new)
                p_next = self.p + self.mix_rate * f_new - correction

            except RuntimeError as e:
                # This was manually raised due to condition number
                if self.verbose:
                    log.info(msg=f"[PDIIS] {e} Falling back to linear mixing.")
                p_next = self.p + self.mix_rate * f_new

            except np.linalg.LinAlgError as e:
                # This is actual numerical failure in np.linalg.solve
                if self.verbose:
                    log.info(msg=f"[PDIIS] np.linalg.solve failed: {e}. Falling back to linear mixing.")
                p_next = self.p + self.mix_rate * f_new
        else:
            if self.verbose:
                log.info(msg=f"[PDIIS] Using linear mixing at iteration {self.iter_count + 1} (not periodic time step or not enough history).")
            p_next = self.p + self.mix_rate * f_new



        # Update state
        self.f = f_new.copy()
        self.p = p_next.copy()
        self.iter_count += 1

        return p_next
    
class BroydenFirstMixer:
    """
    Limited-memory Broyden's First Method ("good Broyden").

    Uses low-rank updates to approximate the Jacobian J (not its inverse).
    Applies Sherman-Morrison-Woodbury formula for efficient inverse computations.

    Attributes:
        alpha (float): Initial mixing parameter (inverse scaling of initial J).
        max_hist (int): Maximum number of stored update pairs (Δx, Δr).
        eps (float): Numerical stability threshold.
        J0 (ndarray): Initial Jacobian approximation (scaled identity).
        dx_hist (list): History of Δx = x_n - x_{n-1}.
        dr_hist (list): History of Δr = r_n - r_{n-1}.
    """

    def __init__(self, shape, max_hist=8, alpha=0.1, eps=1e-12):
        self.alpha = alpha
        self.max_hist = max_hist
        self.eps = eps
        self.reset(shape)

    def reset(self, shape):
        self.iter = 0
        self.x_last = np.zeros(shape)
        self.r_last = np.zeros(shape)
        dim = np.prod(shape)
        self.J0 = np.eye(dim) / self.alpha  # Initial J0 ≈ I/alpha
        self.dx_hist = []  # delta x history
        self.dr_hist = []  # delta r history

    def update(self, x, r):
        x = x.ravel()
        r = r.ravel()

        if self.iter == 0:
            delta = np.linalg.solve(self.J0, r)
            x_new = x - delta
            self.x_last = x.copy()
            self.r_last = r.copy()
            self.iter += 1
            return x_new.reshape(x.shape)

        dx = x - self.x_last
        dr = r - self.r_last

        if len(self.dx_hist) >= self.max_hist:
            self.dx_hist.pop(0)
            self.dr_hist.pop(0)
        self.dx_hist.append(dx)
        self.dr_hist.append(dr)

        J0_dx_list = [self.J0 @ dx for dx in self.dx_hist]
        norm_dx_sq = [np.dot(dx, dx) for dx in self.dx_hist]

        U = np.column_stack([
            (dr - J0_dx) / (ndx + self.eps)
            for dr, J0_dx, ndx in zip(self.dr_hist, J0_dx_list, norm_dx_sq)
        ])  
        V = np.column_stack(self.dx_hist)

        J0_inv_r = self.alpha * r

        M = np.eye(len(self.dx_hist)) + self.alpha * (V.T @ U)
        rhs = V.T @ J0_inv_r
        try:
            y = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            y = np.zeros_like(rhs)

        delta = J0_inv_r - self.alpha * (U @ y)

        x_new = x - delta

        self.x_last = x.copy()
        self.r_last = r.copy()
        self.iter += 1

        return x_new.reshape(self.x_last.shape)


class BroydenSecondMixer:
    """
    Implements Broyden's Second Method (also known as "bad Broyden")
    for accelerating fixed-point iterations such as those arising
    in SCF (self-consistent field) procedures.

    This mixer constructs an approximation to the inverse Jacobian of the residual
    using a low-rank update formula and applies it to iteratively improve convergence.

    The method uses limited-memory rank-1 updates:
        x_{n+1} = x_n - B_n * r_n
    where B_n ≈ J^{-1} is the inverse Jacobian built from the update history.

    Attributes:
        alpha (float): Initial mixing parameter for the first step.
        max_hist (int): Maximum number of update pairs (u, r) stored for low-rank updates.
        eps (float): Threshold to avoid numerical instability in inner products.
        B0 (ndarray): Initial inverse Jacobian approximation (scaled identity).
        u_hist (list): History of update vectors u_n = s_n - B_n * delta_r_n.
        r_hist (list): History of delta_r_n = r_n - r_{n-1}.
    """

    def __init__(self, shape, max_hist=8, alpha=0.1, eps=1e-12):
        self.alpha = alpha        # Initial mixing factor
        self.max_hist = max_hist  # Max number of correction terms
        self.eps = eps            # Numerical stability threshold
        self.reset(shape)

    def reset(self, shape):
        self.iter = 0
        self.x_last = np.zeros(shape)
        self.r_last = np.zeros(shape)
        dim = np.prod(shape)
        self.B0 = -self.alpha * np.eye(dim)  # Initial inverse Jacobian guess
        self.u_hist = []  # History of update vectors u_n = s_n - B delta_r
        self.r_hist = []  # Corresponding delta_r vectors

    def update(self, x, r):
        """
        Perform one Broyden update step: x_{n+1} = x_n - B_n * r_n.

        This function applies the approximate inverse Jacobian B_n
        to the current residual r_n to compute the next guess x_{n+1}.
        The internal approximation B_n is updated based on the history
        of residual differences and solution updates.

        Args:
            x (np.ndarray): Current solution guess (arbitrary shape).
            r (np.ndarray): Residual vector at the current guess.

        Returns:
            np.ndarray: Updated solution guess (same shape as input x).
        """
        x = x.ravel()
        r = r.ravel()

        if self.iter == 0:
            x_new = x - self.B0 @ r
            self.x_last = x.copy()
            self.r_last = r.copy()
            self.iter += 1
            return x_new.reshape(self.x_last.shape)

        # Step 1: Compute s_n = x - x_last, delta_r = r - r_last
        s_n = x - self.x_last
        delta_r = r - self.r_last

        # Step 2: Build B * delta_r incrementally
        B_delta_r = self.B0 @ delta_r
        for u_j, r_j in zip(self.u_hist, self.r_hist):
            rj_dot = np.dot(r_j, delta_r)
            norm2 = np.dot(r_j, r_j)
            if norm2 > self.eps:
                B_delta_r += u_j * (rj_dot / (norm2 + self.eps))

        # Step 3: u_n = s_n - B delta_r (corrected formula)
        u_n = s_n - B_delta_r

        # Step 4: Truncate history if needed
        if len(self.u_hist) >= self.max_hist:
            self.u_hist.pop(0)
            self.r_hist.pop(0)
        self.u_hist.append(u_n)
        self.r_hist.append(delta_r)

        # Step 5: Apply B_n * r using low-rank update
        B_r = self.B0 @ r
        for u_j, r_j in zip(self.u_hist, self.r_hist):
            rj_dot = np.dot(r_j, r)
            norm2 = np.dot(r_j, r_j)
            if norm2 > self.eps:
                B_r += u_j * (rj_dot / (norm2 + self.eps))

        # Step 6: Final update
        x_new = x - B_r

        # Step 7: Cache for next iteration
        self.x_last = x.copy()
        self.r_last = r.copy()
        self.iter += 1

        return x_new.reshape(self.x_last.shape)

