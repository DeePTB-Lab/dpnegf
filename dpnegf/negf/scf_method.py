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

# class Block_BroydenFirstMixer:
#     """
#     Efficient Broyden's First Method (good Broyden) using the Sherman-Morrison-Woodbury formula.

#     Attributes:
#         alpha (float): Initial mixing parameter (J0 = I/alpha).
#         eps (float): Numerical stability threshold.
#     """
#     def __init__(self, init_x, alpha:float=0.1, k:int=None):
        
#         assert init_x.ndim == 1, "init_x must be a 1D array"
#         self.init_x = init_x
#         self.alpha = alpha
#         self.beta = 1 # Adaptive mixing factor
#         self.eps = 1e-12  # Numerical stability threshold

#         if k is None:
#             self.k = int(len(init_x)/10)
#         else:
#             assert k > 0, "k must be a positive integer"
#             assert k <= len(init_x), "k must be less than or equal to the length of init_x"
#             self.k = k

        
#         self.reset(init_x.shape)
        
#     def reset(self, shape):
#         self.iter = 0
#         self.x_n = np.zeros(shape)
#         self.x_nm1 = np.zeros(shape)
#         self.dim = np.prod(shape)
#         self.shape = shape
#         self.J0 = -np.eye(self.dim) / self.alpha  # Jacobian approximation
#         self.Bm = np.zeros_like(self.J0)  # Inverse Jacobian

#     @staticmethod
#     def get_partial_jacobian(df, dx, column_indices, eps=1e-12):
    
#         assert df.ndim == 1, "df must be a 1D array"
#         assert dx.ndim == 1, "dx must be a 1D array"
#         assert df.shape == dx.shape, "df and dx must have the same shape"

#         dx_norm_sq = np.dot(dx, dx) + eps  # scalar
#         d_outer = np.outer(df, dx) / dx_norm_sq  # full rank-1 Jacobian approx (d x d)

#         # Select only the desired columns
#         return d_outer[:, column_indices]


#     def update(self, f):
    
#         linear_warm_range = 3  # Number of iterations to use linear mixing before switching to Broyden's method

#         if self.iter == 0:
#             x_new = self.init_x + self.alpha * f  
#             self.Bm = -np.eye(self.dim) * self.alpha  # Initial inverse Jacobian
#             self.x_nm1 = self.init_x.copy()
        
#         elif self.iter < linear_warm_range:
#             x_new = self.x_n + self.alpha * f  # Linear mixing for first few iterations
#             self.Bm = -np.eye(self.dim) * self.alpha  # Reset inverse Jacobian
#             self.x_nm1 = self.x_n.copy()  # Store previous x

#         else:
#             dx = self.x_n - self.x_nm1  
#             df = f - self.f_last

#             x_new = self.x_n -self.Bm @ f 

#             # Randomly select k directions
#             rng = np.random.default_rng()
#             idx_seq = rng.choice(len(self.init_x), size=self.k, replace=False)
#             U = np.eye(len(self.init_x))[:, idx_seq] # Randomly select k directions

#             # Compute the partial Jacobian using finite differences
#             J_U = self.get_partial_jacobian(df, dx, idx_seq, eps=self.eps)  # Compute partial Jacobian
#             Bm_J_U = self.Bm @ J_U  # Apply the current inverse Jacobian to the partial Jacobian
#             UT_Bm_J_U = U.T @ Bm_J_U  # Compute U^T * B_m * J_U
#             M = np.linalg.solve(UT_Bm_J_U, U.T @ self.Bm) 
#             # Update the inverse Jacobian using the Sherman-Morrison-Woodbury formula
#             self.Bm = self.Bm - (Bm_J_U - U) @ M

#             self.x_nm1 = self.x_n.copy()  # Store previous x

#         # Update state
#         self.x_n = x_new.copy() 
#         self.f_last = f.copy()  
#         self.iter += 1

#         return x_new
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
        H_r = self.B0 @ r
        for u_j, r_j in zip(self.u_hist, self.r_hist):
            rj_dot = np.dot(r_j, r)
            norm2 = np.dot(r_j, r_j)
            if norm2 > self.eps:
                H_r += u_j * (rj_dot / (norm2 + self.eps))

        # Step 6: Final update
        x_new = x - H_r

        # Step 7: Cache for next iteration
        self.x_last = x.copy()
        self.r_last = r.copy()
        self.iter += 1

        return x_new.reshape(self.x_last.shape)


class AndersonMixer:
    """
    AndersonMixer implements Anderson mixing for accelerating fixed-point iterations,
    commonly used in self-consistent field (SCF) calculations.
    Attributes:
        m (int): Number of history steps to retain for mixing.
        alpha (float): Mixing parameter for linear mixing (0 < alpha <= 1).
        verbose (bool): If True, prints internal details for debugging.
        dx_hist (list): History of differences in input vectors (xk - x_{k-1}).
        df_hist (list): History of differences in function outputs (f(xk) - f(x_{k-1})).
        first_three (bool): Flag to handle the first three iterations with linear mixing.
        iter (int): Iteration counter.
        xkm1 (np.ndarray or None): Previous input vector x_{k-1}.
        fkm1 (np.ndarray or None): Previous function output f(x_{k-1}).
        beta (float): Damping factor for the Anderson update.
    """
    def __init__(self, m:int=5, alpha:float=0.2, beta:float=1, verbose=False):
        """
        Initializes the SCF method parameters.

        Args:
            m (int, optional): Number of previous iterations to store for history-based methods. Defaults to 5.
            alpha (float, optional): Mixing parameter or step size for the update. Defaults to 0.1.
            verbose (bool, optional): If True, enables verbose output for debugging or logging. Defaults to False.

        Attributes:
            dx_hist (list): History of differences between consecutive x values (x_k - x_{k-1}).
            df_hist (list): History of differences between consecutive function values (f(x_k) - f(x_{k-1})).
            first_three (bool): Flag to handle the first three iterations separately.
            iter (int): Iteration counter.
            xkm1: Previous x value (x_{k-1}).
            fkm1: Previous function value (f(x_{k-1})).
            beta (float): Damping factor for the update, initialized to 1.
        """
        self.m = m
        self.alpha = alpha
        self.verbose = verbose
        self.dx_hist = []  #  xk - x_{k-1}
        self.df_hist = []  #  f(x_k) - f(x_{k-1})
        self.first_three = True  # Flag to handle first three iterations separately
        self.iter = 0  # Iteration counter
        self.xkm1 = None  # x_{k-1}
        self.fkm1 = None  # f(x_{k-1})

        self.beta = beta # damping factor (0 < beta <= 1) for the Anderson update, default is 1 (no damping)

    def reset(self):
        """
        Resets the internal state of the SCF method.
        This method clears the history of variable and function differences,
        resets iteration counters, and sets previous step variables to None,
        preparing the object for a fresh SCF cycle.
        """
        
        self.dx_hist.clear()
        self.df_hist.clear()
        self.first_three = True
        self.iter = 0
        self.xkm1 = None  # Reset previous x_{k-1}
        self.fkm1 = None  # Reset previous f(x_{k-1})

    def update(self, fk, xk):
        """
        Update the current estimate using Anderson mixing or linear mixing.
        Parameters
        ----------
        fk : np.ndarray
            The current function value (e.g., residual or fixed-point map at xk).
        xk : np.ndarray
            The current estimate of the solution.
        Returns
        -------
        np.ndarray
            The updated estimate after applying Anderson or linear mixing.
        Raises
        ------
        AssertionError
            If `fk` or `xk` are not numpy arrays or if their shapes do not match.
        Notes
        -----
        - For the first three iterations, linear mixing is used.
        - After the first three iterations, Anderson mixing is applied using the history of previous steps.
        - If the Anderson mixing matrix is rank-deficient, falls back to linear mixing.
        """
        # Ensure fk and xk are numpy arrays and have the same shape
        assert isinstance(fk, np.ndarray), "fk must be a numpy array"
        assert isinstance(xk, np.ndarray), "xk must be a numpy array"
        assert fk.shape == xk.shape, "fk and xk must have the same shape"

        if self.first_three:
            if self.iter < 3:
                # self.dx_list.append(dx.copy())
                # self.df_hist.append(df.copy())
                self.iter += 1
                x_new = xk + self.alpha * (fk - xk)  # Linear mixing for first three iterations
                self.xkm1 = xk.copy()  # Store x_k for next iteration
                self.fkm1 = fk.copy()
                return x_new  # linear mixing
            else:
                self.first_three = False
            

        dx = xk - self.xkm1  # dx = x_k - x_{k-1}
        df = fk - self.fkm1  # df = f(x_k) - f(x_{k-1})
        self.xkm1 = xk.copy()  # Store x_k for next iteration
        self.fkm1 = fk.copy()  # Store f_k for next iteration            

        # Keep only last m entries
        if len(self.dx_hist) >= self.m:
            self.dx_hist.pop(0)
            self.df_hist.pop(0)

        self.dx_hist.append(dx.copy())
        self.df_hist.append(df.copy())

        # Construct matrix R = [r1, r2, ..., rn]
        
        try:
            Gk = np.column_stack([df_i - dx_i for df_i, dx_i in zip(self.df_hist, self.dx_hist)])# [... gk-1 - gk-2, gk - gk-1 ]
            c = np.linalg.lstsq(Gk, (fk-xk), rcond=None)[0] # Solve least squares: min ||Gk @ c - (fk-xk)||
            correction = sum(c_i * df_i for c_i, df_i in zip(c, self.df_hist))
            xkp1 =  xk + self.beta * ((fk - xk) - correction) # Update x_k+1

        except np.linalg.LinAlgError:
            # Fallback to linear mixing if R is rank-deficient
            log.info("[Anderson] Linear algebra error, fallback to linear mixing.")
            xkp1 = xk + self.alpha * (fk - xk)

        self.iter += 1
        return xkp1.reshape(fk.shape)

