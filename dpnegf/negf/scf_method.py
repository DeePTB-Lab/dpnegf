import numpy as np
import logging

log = logging.getLogger(__name__)


class DIISMixer:
    """
    DIISMixer implements the Direct Inversion in the Iterative Subspace (DIIS) method 
    for accelerating self-consistent field (SCF) convergence.
    max_hist : int, optional
        Maximum number of history vectors to store for DIIS extrapolation (default: 6).
    alpha : float, optional
        Mixing parameter for linear mixing during warmup (default: 0.2).
    linear_warmup : int, optional
        Number of initial iterations to use linear mixing before switching to DIIS (default: 1).
    Attributes
    x_hist : list of ndarray
        History of previous input vectors (e.g., potentials).
    r_hist : list of ndarray
        History of previous residual vectors.
    iter : int
        Iteration counter.
    max_hist : int
        Maximum number of history vectors.
    alpha : float
        Linear mixing parameter.
    linear_warmup : int
        Number of linear mixing iterations before DIIS.
    """
    def __init__(self, max_hist:int=6, alpha:float=0.2, linear_warmup:int=1):
        self.max_hist = max_hist
        self.alpha = alpha
        self.linear_warmup = linear_warmup  # Number of iterations to use linear mixing before DIIS
        
        self.x_hist = []
        self.r_hist = []
        self.iter = 0  # Iteration counter

    def reset(self):
        """
        Reset the internal state of the mixer by clearing the history of x and r,
        setting the iteration counter to zero, and logging the reset action.
        """
        self.x_hist.clear()
        self.r_hist.clear()
        self.iter = 0
        log.info("[DIIS] Mixer state reset.")

    def update(self, x_new:np.ndarray, r_new:np.ndarray):
        """
        Update the current solution vector using DIIS (Direct Inversion in the Iterative Subspace) or linear mixing.

        This method manages the iterative update of a solution vector `x_new` and its corresponding residual `r_new`.
        It stores a history of previous solution and residual vectors up to `max_hist` entries. During the initial
        `linear_warmup` iterations, it applies linear mixing. After that, it constructs the DIIS B matrix and solves
        for optimal mixing coefficients to accelerate convergence. If the B matrix is singular, it falls back to 
        linear mixing.

        Parameters
        ----------
        x_new : np.ndarray
            The new solution vector to be mixed.
        r_new : np.ndarray
            The new residual vector.

        Returns
        -------
        np.ndarray
            The updated (mixed) solution vector.
        """
        self.iter += 1

        x_new = x_new.copy().reshape(-1,1)
        r_new = r_new.copy().reshape(-1,1)

        # Store new history
        if len(self.x_hist) >= self.max_hist:
            self.x_hist.pop(0)
            self.r_hist.pop(0)

        self.x_hist.append(x_new)
        self.r_hist.append(r_new)

        if self.iter <= self.linear_warmup : # The first iteration
            x_new = (x_new - r_new) + self.alpha * r_new  # Linear mixing
            return x_new.ravel() # Not enough history yet

        # Construct B matrix (r_i · r_j)
        n = len(self.r_hist)
        B = np.empty((n + 1, n + 1))
        B[-1, :] = 1
        B[:, -1] = -1
        B[-1, -1] = 0
        for i in range(n):
            for j in range(n):
                B[i, j] = np.dot(self.r_hist[i].T, self.r_hist[j])

        # Right-hand side of linear system
        rhs = np.zeros(n + 1)
        rhs[-1] = 1
        rhs = rhs.reshape(-1, 1)

        try:
            coeffs = np.linalg.solve(B, rhs)[:-1]  # drop Lagrange multiplier
        except np.linalg.LinAlgError:
            log.warning("[DIIS] Singular matrix in DIIS mixing. Falling back to lienar mixing.")
            # Fallback to linear mixing if B is singular
            x_new = (x_new - r_new) + self.alpha * r_new  # Linear mixing
            return x_new.ravel()

        # Construct mixed x
        x_mixed = sum(c * x for c, x in zip(coeffs, self.x_hist))
        return x_mixed.ravel()  # Return as 1D array


class PDIISMixer:
    """
    PDIISMixer implements the Periodic Direct Inversion in the Iterative Subspace (PDIIS) 
    mixing scheme for accelerating self-consistent field (SCF) convergence.

    init_f : np.ndarray
        Initial state (e.g., potential or density) for the mixer.
    mix_rate : float, optional
        Linear mixing rate (default: 0.2).
    max_hist : int, optional
        Maximum number of history vectors to store for DIIS extrapolation (default: 4).
    mixing_period : int, optional
        Number of iterations between DIIS mixing steps (default: 2).
    Attributes
    mix_rate : float
        Linear mixing rate.
    max_hist : int
        Maximum number of history vectors.
    mixing_period : int
        Number of iterations between DIIS mixing steps.
    iter_count : int
        Current iteration count.
    x : np.ndarray or None
        Current mixed state.
    x_last : np.ndarray or None
        Previous mixed state.
    f : np.ndarray or None
        Current input state.
    R : list of np.ndarray
        History of differences in mixed states.
    F : list of np.ndarray
        History of differences in input states.
    Methods
    reset(new_init_f=None)
        Reset the mixer, optionally with a new initial state.
    update(f_new)
        Perform one PDIIS mixing update based on the new input state.
    Notes
    -----
    - The mixer alternates between linear mixing and DIIS mixing according to `mixing_period`.
    - If the DIIS matrix is ill-conditioned or a numerical error occurs, the mixer falls back to linear mixing.
    """
    def __init__(self, init_x, mix_rate=0.2, max_hist=4, mixing_period=2):
        assert isinstance(init_x, np.ndarray), "init_x must be a numpy array"
        
        self.mix_rate = mix_rate
        self.max_hist = max_hist
        self.mixing_period = mixing_period
        
        self.iter_count = 1
        self.x = init_x.copy() # x_i
        self.x_last = None  # x_{i-1}
        self.f = None
        self.R = []
        self.F = []

    def reset(self, new_init_f=None):
        """Reset the mixer, optionally with a new initial potential."""
        self.iter_count = 1
        self.x = None
        self.f = None
        self.R = []
        self.F = []
        if new_init_f is not None:
            assert isinstance(new_init_f, np.ndarray), "new_init_f must be a numpy array"
            self.f = new_init_f.copy()

    def update(self, f_new):
        """
        Perform one PDIIS mixing update based on the new input f_new.

        Parameters
        ----------
        f_new : np.ndarray
            Newly computed state (e.g., electrostatic potential).

        Returns
        -------
        x_next : np.ndarray
            The next mixed state.
        """
        assert isinstance(f_new, np.ndarray), "f_new must be a numpy array"
        
        f_new = f_new.copy()


        if self.iter_count <= 2:
            # First two iteration
            x_next = self.x + self.mix_rate * (f_new - self.x)  # Linear mixing
            if self.iter_count == 2:
                self.x_last = self.x.copy()
            self.x = x_next.copy()
            self.f = f_new.copy()  # Update current state
            self.iter_count += 1
            return x_next  # Return the mixed state

        else: # After the first two iterations, PDIIS can be used
            assert f_new.shape == self.f.shape, "Shape mismatch in x_new and current state"
            dx_i = self.x - self.x_last # dx_i = x_i - x_{i-1}
            df_i = f_new - self.f  # df_i = f_i - f_{i-1}

            # Store new history
            if len(self.F) >= self.max_hist:
                self.R.pop(0)
                self.F.pop(0)
            self.R.append(dx_i)  # Store the difference in potentials
            self.F.append(df_i)

            do_pdiis = self.iter_count  % self.mixing_period == 0
            x_next = None # x_{i+1}
            if do_pdiis:
                log.info(msg=f"[PDIIS] Performing DIIS mixing at iter {self.iter_count}")
                F_mat = np.column_stack(self.F)
                R_mat = np.column_stack(self.R)
                FtF = F_mat.T @ F_mat

                try:
                    cond_FtF = np.linalg.cond(FtF)
                    if cond_FtF > 1e10:
                        log.warning(msg=f"[PDIIS] Warning: FtF matrix condition number ({cond_FtF:.2e}) is too high. Skipping DIIS.")
                        log.warning(f"[PDIIS DEBUG] cond(FtF) = {cond_FtF:.2e}")
                        log.warning(f"[PDIIS DEBUG] Norms of F vectors: {[np.linalg.norm(f) for f in self.F]}")
                        log.warning(f"[PDIIS DEBUG] Rank of F_mat: {np.linalg.matrix_rank(F_mat)}")
                        raise RuntimeError("Ill-conditioned FtF matrix in PDIIS")

                    correction = (R_mat + self.mix_rate * F_mat) @ np.linalg.solve(FtF, F_mat.T @ f_new)
                    x_next = self.x + self.mix_rate * f_new - correction
                except RuntimeError as e:
                    # This was manually raised due to condition number
                    log.warning(msg=f"[PDIIS] {e} Falling back to linear mixing.")
                    x_next = self.x + self.mix_rate * (f_new - self.x) 

                except np.linalg.LinAlgError as e:
                    # Numerical failure in np.linalg.solve
                    log.warning(msg=f"[PDIIS] np.linalg.solve failed: {e}. Falling back to linear mixing.")
                    x_next = self.x + self.mix_rate * (f_new - self.x) 

            else:
                log.info(msg=f"[PDIIS] Using linear mixing at iteration {self.iter_count} (not periodic time step).")
                x_next = self.x + self.mix_rate * (f_new - self.x)  # Linear mixing

            # Update state
            self.f = f_new.copy()
            self.x_last = self.x.copy()
            self.x = x_next.copy()
            self.iter_count += 1

            return x_next
    



class BroydenFirstMixer:
    """
    Implements the first Broyden mixing method for accelerating self-consistent field (SCF) iterations.

    Attributes:
        init_x (np.ndarray): Initial guess for the variable to be mixed.
        alpha (float): Linear mixing parameter (default: 0.1).
        beta (float): Adaptive mixing factor (currently unused, default: 1).
        eps (float): Numerical stability threshold for denominator (default: 1e-12).
        iter (int): Current iteration count.
        x_n (np.ndarray): Current value of the variable.
        x_nm1 (np.ndarray): Previous value of the variable.
        dim (int): Flattened dimension of the variable.
        shape (tuple): Shape of the variable.
        J0 (np.ndarray): Initial Jacobian approximation.
        J_inv (np.ndarray): Current inverse Jacobian approximation.
        f_last (np.ndarray): Last residual vector.
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
        """
        Update the solution vector using a combination of linear mixing and Broyden's method.
        For the first few iterations(warm up), it uses simple linear mixing to stabilize convergence.
        After a specified number of iterations, it switches to Broyden's first method to accelerate convergence 
        by approximating the inverse Jacobian.
        Args:
            f (np.ndarray): The current residual or function value at the current solution vector.
        Returns:
            np.ndarray: The updated solution vector after applying the mixing or Broyden's update.
        """

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
        x_{n+1} = x_n - H_n * f_n
    where H_n ≈ J^{-1} is the inverse Jacobian.

    Attributes:
        alpha (float): Initial mixing parameter for the first step.
        eps (float): Threshold to avoid numerical instability in inner products.
        H0 (ndarray): Initial inverse Jacobian approximation (scaled identity).
        df_hist (list): History of delta_df_n = f_n - f_{n-1}.
    """

    def __init__(self, shape, alpha=0.1, eps=1e-12):
        self.alpha = alpha        # Initial mixing factor
        self.eps = eps            # Numerical stability threshold
        self.reset(shape)

    def reset(self, shape):
        self.iter = 0
        self.x_last = np.zeros(shape)
        self.f_last = np.zeros(shape)
        dim = np.prod(shape)
        self.H0 = -self.alpha * np.eye(dim)  # Initial inverse Jacobian guess
        self.df_hist = []  # Corresponding delta_r vectors

    def update(self, x, f):
        """
        Perform one Broyden update step: x_{n+1} = x_n - H_n * f_n.

        This function applies the approximate inverse Jacobian H_n
        to the current residual f_n to compute the next guess x_{n+1}.
        The internal approximation B_n is updated based on the history
        of residual differences and solution updates.

        Args:
            x (np.ndarray): Current solution guess (arbitrary shape).
            f (np.ndarray): Residual vector at the current guess.

        Returns:
            np.ndarray: Updated solution guess (same shape as input x).
        """
        x = x.reshape(-1,1)
        f = f.reshape(-1,1)

        if self.iter == 0:
            x_new = x - self.H0 @ f
            self.x_last = x.copy()
            self.f_last = f.copy()
            self.Hnm1 = self.H0.copy()  # Initial inverse Jacobian
            self.iter += 1
            return x_new.ravel()

        # Step 1: Compute s_n = x - x_last, delta_r = r - r_last
        dx = x - self.x_last
        df = f - self.f_last

        u_n = dx - self.Hnm1 @ df  # Update vector
        norm_df = np.dot(df.T, df)
        Hn = self.Hnm1 + np.outer(u_n, df) / (norm_df + self.eps)  # Update inverse Jacobian
        x_new = x - Hn @ f  # Compute new solution guess
        self.Hnm1 = Hn  # Update the last inverse Jacobian

        # Step 7: Cache for next iteration
        self.x_last = x.copy()
        self.f_last = f.copy()
        self.iter += 1

        return x_new.ravel()


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
    def __init__(self, m:int=5, alpha:float=0.2, beta:float=1, num_linear_warmup:int=3 , verbose=False):
        """
        Initializes the SCF method parameters.

        Args:
            m (int, optional): Number of previous iterations to store for history-based methods. Defaults to 5.
            alpha (float, optional): Mixing parameter or step size for the update. Defaults to 0.1.
            verbose (bool, optional): If True, enables verbose output for debugging or logging. Defaults to False.

        Attributes:
            dx_hist (list): History of differences between consecutive x values (x_k - x_{k-1}).
            df_hist (list): History of differences between consecutive function values (f(x_k) - f(x_{k-1})).
            first_linear (bool): Flag to handle the first linear mixing iterations separately.
            num_linear_warmup (int): Number of iterations to use linear mixing before switching to Anderson mixing.
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
        self.first_linear = True  # Flag to handle first three iterations separately
        self.num_linear_warmup = num_linear_warmup  # Number of iterations to use linear mixing before switching to Anderson mixing
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
        self.first_linear = True
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

        if self.first_linear:
            if self.iter < self.num_linear_warmup:
                xkp1 = xk + self.alpha * (fk - xk)  # Linear mixing for first three iterations
                if self.iter > 0:
                    dx = xk - self.xkm1
                    df = fk - self.fkm1
                    self.dx_hist.append(dx.copy())
                    self.df_hist.append(df.copy())
                self.xkm1 = xk.copy()  # Store x_k for next iteration
                self.fkm1 = fk.copy()
                self.iter += 1
                return xkp1  # linear mixing
            else:
                self.first_linear = False
            

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
            log.warning("[Anderson] Linear algebra error, fallback to linear mixing.")
            xkp1 = xk + self.alpha * (fk - xk)

        self.iter += 1
        return xkp1.reshape(fk.shape)

