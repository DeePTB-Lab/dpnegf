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
    Efficient Broyden's First Method (good Broyden) using the Sherman-Morrison-Woodbury formula.

    Attributes:
        alpha (float): Initial mixing parameter (J0 = I/alpha).
        eps (float): Numerical stability threshold.
    """
    def __init__(self, init_x, alpha=0.1):
        self.init_x = init_x
        self.alpha = alpha
        self.beta = 1 # Adaptive mixing factor
        self.reset(init_x.shape)
        
        self.eps = 1e-12  # Numerical stability threshold

    def reset(self, shape):
        self.iter = 0
        self.x_n = np.zeros(shape)
        self.x_nm1 = np.zeros(shape)
        self.dim = np.prod(shape)
        self.shape = shape
        self.J0 = -np.eye(self.dim) / self.alpha  # Jacobian approximation
        self.J_inv = np.zeros_like(self.J0)  # Inverse Jacobian


    def update(self, f):

        linear_warm_range = 3  # Number of iterations to use linear mixing before switching to Broyden's method

        if self.iter == 0:
            x_new = self.init_x + self.alpha * f  
            self.J_inv = -np.eye(self.dim) * self.alpha  # Initial inverse Jacobian
            self.x_nm1 = self.init_x.copy()
        
        elif self.iter < linear_warm_range:
            x_new = self.x_n + self.alpha * f  # Linear mixing for first few iterations
            self.J_inv = -np.eye(self.dim) * self.alpha  # Reset inverse Jacobian
            self.x_nm1 = self.x_n.copy()  # Store previous x

        else:
            dx = self.x_n - self.x_nm1  
            df = f - self.f_last

            dx = dx.reshape(-1, 1)  # Ensure dx is a column vector
            df = df.reshape(-1, 1)  # Ensure df is a column vector

            # df_norm = np.linalg.norm(df)
            # if self.iter == linear_warm_range:
            #     self.last_df_norm = df_norm
            #     self.beta = 1.0  # Initial beta value for adaptive mixing
            # else:
            #     if df_norm > self.last_df_norm:
            #         self.beta = max(0.1, self.beta * 0.5)
            #     else:
            #         self.beta = min(1.0, self.beta * 1.2)
            # self.last_df_norm = df_norm

            J_inv_df = self.J_inv @ df  # J^{-1} * df
            numerator = (dx - J_inv_df) @ (dx.T @ self.J_inv)  # (dim,1) @ (1, dim) = (dim, dim)
            denominator = dx.T @ J_inv_df  + self.eps  # (1, dim) @ (dim, 1) = (1, 1) 
            self.J_inv = self.J_inv + numerator / denominator
            x_new = self.x_n -self.J_inv @ f  # Update x using the new inverse Jacobian
            self.x_nm1 = self.x_n.copy()  # Store previous x

        # Update state
        self.x_n = x_new.copy() 
        self.f_last = f.copy()  
        self.iter += 1

        return x_new

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

