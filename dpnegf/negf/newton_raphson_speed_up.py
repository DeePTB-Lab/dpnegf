import time
import unittest
from typing import Optional, Tuple, List, Callable

import numpy as np
from scipy.sparse import lil_matrix

def _impose_j_bound(inout, nx, ny, nz, typ, val, mask):
    '''impose the special mask for boundary points'''
    my_boundary_types_ = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'Dirichlet']
    my_displ_ = [+1, -1, +nx, -nx, +nx*ny, -nx*ny]
    for t, d in zip(my_boundary_types_, my_displ_):
        # perf bottle-neck: search
        i = np.where(np.array(typ) == t)[0] # all indices of the boundary type
        assert len(i) > 0, f'no grid point with type {t} found'
        # scipy sparse matrix also supports the parallized assignment
        inout[i, i] = val
        inout[i, i + d] = mask

    # there is also a special type "Dirichlet" for the Dirichlet boundary condition
    i = np.where(typ == 'Dirichlet')[0]
    inout[i, i] = val

class TestImposeJacobianBoundaryPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e3
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        # say we have 6 boundary types, each has 300 points
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten()
        
    def test_impose_bound(self):
        # create a dummy inout matrix
        ref = lil_matrix((self.nr, self.nr), dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz
        t = time.time()
        
        for i in range(self.nr):
            if self.typ[i] == "xmin":
                ref[i, i] = -1.0
                ref[i, i + 1] = 1.0
            elif self.typ[i] == "xmax":
                ref[i, i] = -1.0
                ref[i, i - 1] = 1.0
            elif self.typ[i] == "ymin":
                ref[i, i] = -1.0
                ref[i, i + nx] = 1.0
            elif self.typ[i] == "ymax":
                ref[i, i] = -1.0
                ref[i, i - nx] = 1.0
            elif self.typ[i] == "zmin":
                ref[i, i] = -1.0
                ref[i, i + nx*ny] = 1.0
            elif self.typ[i] == "zmax":
                ref[i, i] = -1.0
                ref[i, i - nx*ny] = 1.0
        print(f'old boundary impose method took {time.time() - t:.4f} seconds')
        
        inout = lil_matrix((self.nr, self.nr), dtype=float)
        t = time.time()
        # call the function
        _impose_j_bound(inout, nx, ny, nz, self.typ, -1.0, 1.0)
        print(f'_impose_j_bound took {time.time() - t:.4f} seconds')
    
        # check if the inout matrix is equal to the reference matrix
        # check all diagonal elements
        self.assertTrue(all(inout[i, i] == ref[i, i] for i in range(self.nr)))
        # check xmin points
        ind_xmin = np.where(self.typ == 'xmin')[0]
        self.assertTrue(all(inout[i, i + 1] == ref[i, i + 1] for i in ind_xmin))
        # check xmax points
        ind_xmax = np.where(self.typ == 'xmax')[0]
        self.assertTrue(all(inout[i, i - 1] == ref[i, i - 1] for i in ind_xmax))
        # check ymin points
        ind_ymin = np.where(self.typ == 'ymin')[0]
        self.assertTrue(all(inout[i, i + nx] == ref[i, i + nx] for i in ind_ymin))
        # check ymax points
        ind_ymax = np.where(self.typ == 'ymax')[0]
        self.assertTrue(all(inout[i, i - nx] == ref[i, i - nx] for i in ind_ymax))
        # check zmin points
        ind_zmin = np.where(self.typ == 'zmin')[0]
        self.assertTrue(all(inout[i, i + nx*ny] == ref[i, i + nx*ny] for i in ind_zmin))
        # check zmax points
        ind_zmax = np.where(self.typ == 'zmax')[0]
        self.assertTrue(all(inout[i, i - nx*ny] == ref[i, i - nx*ny] for i in ind_zmax))

def _impose_b_bound(inout, nx, ny, nz, typ, phi, dirichlet_pot):
    '''impose the special mask for boundary points in b vector'''
    my_boundary_types_ = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'Dirichlet']
    my_displ_ = [+1, -1, +nx, -nx, +nx*ny, -nx*ny]
    for t, d in zip(my_boundary_types_, my_displ_):
        # perf bottle-neck: search
        i = np.where(typ == t)[0]
        assert len(i) > 0, f'no grid point with type {t} found'
        inout[i] = phi[i] - phi[i + d]
        
    # there is also a special type "Dirichlet" for the Dirichlet boundary condition
    i = np.where(typ == 'Dirichlet')[0]
    inout[i] = phi[i] - dirichlet_pot[i]

class TestImposeRHSVecBoundaryPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e3
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten()
        self.phi = np.random.rand(self.nr).astype(float)
        self.fixed_pot = np.random.rand(self.nr).astype(float)
        
    def test_impose_bound(self):
        # create a dummy inout vector
        ref = np.zeros(self.nr, dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz
        t = time.time()
        
        for i in range(self.nr):
            if self.typ[i] == "xmin":
                ref[i] = self.phi[i] - self.phi[i + 1]
            elif self.typ[i] == "xmax":
                ref[i] = self.phi[i] - self.phi[i - 1]
            elif self.typ[i] == "ymin":
                ref[i] = self.phi[i] - self.phi[i + nx]
            elif self.typ[i] == "ymax":
                ref[i] = self.phi[i] - self.phi[i - nx]
            elif self.typ[i] == "zmin":
                ref[i] = self.phi[i] - self.phi[i + nx*ny]
            elif self.typ[i] == "zmax":
                ref[i] = self.phi[i] - self.phi[i - nx*ny]
            elif self.typ[i] == "Dirichlet":
                ref[i] = self.phi[i] - self.fixed_pot[i]
        print(f'old boundary impose method took {time.time() - t:.4f} seconds')
        
        inout = np.zeros(self.nr, dtype=float)
        t = time.time()
        # call the function
        _impose_b_bound(inout, nx, ny, nz, self.typ, self.phi, self.fixed_pot)
        print(f'_impose_b_bound took {time.time() - t:.4f} seconds')
    
        # check if the inout vector is equal to the reference vector
        self.assertTrue(np.all(inout == ref))

def coloumb(chr: float|np.ndarray) -> float|np.ndarray:
    '''convert the charge to unit of Coulomb'''
    assert isinstance(chr, (float, np.ndarray)), "chr must be a float or numpy array"
    from dpnegf.utils.constants import elementary_charge as e
    return chr * e

def _bflux_impl(jflux, i, nx, ny, nz, phi, full_size=True) -> np.ndarray:
    '''calculate the b vector from the jflux and phi'''
    mydisp_ = [-1, +1, -nx, +nx, -nx*ny, +nx*ny]
    if full_size:
        assert jflux.shape == (6, len(phi))
        bflux = np.zeros((6, len(phi)), dtype=float)
        for j, d in enumerate(mydisp_):
            bflux[j, i] = jflux[j, i] * (phi[i + d] - phi[i])
    else:
        assert jflux.shape == (6, len(i))
        bflux = np.array([jf * (phi[i + d] - phi[i]) for jf, d in zip(jflux, mydisp_)])
        assert bflux.shape == (6, len(i))
    return bflux

class TestBFluxImplPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e3
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        # set the boundary types
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten()
        self.r = np.random.rand(self.nr, 3).astype(float)
        self.sigma = np.random.rand(self.nr, 3).astype(float)
        self.eps = np.random.rand(self.nr).astype(float)
        self.eps0 = 8.854187817e-12
        self.avgeps = lambda eps1, eps2: (eps1 + eps2) / 2.0  # arithmetic mean

    def test_bflux_impl(self):
        ind = np.where(self.typ == "in")[0]
        jflux = np.random.rand(6, self.nr).astype(float)
        phi = np.random.rand(self.nr).astype(float)
        # calculate the reference result
        ref = np.zeros((6, self.nr), dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz
        t = time.time()
        for i in range(self.nr):
            if self.typ[i] == 'in':
                ref[0, i] = jflux[0, i] * (phi[i     - 1] - phi[i])
                ref[1, i] = jflux[1, i] * (phi[i     + 1] - phi[i])
                ref[2, i] = jflux[2, i] * (phi[i    - nx] - phi[i])
                ref[3, i] = jflux[3, i] * (phi[i    + nx] - phi[i])
                ref[4, i] = jflux[4, i] * (phi[i - nx*ny] - phi[i])
                ref[5, i] = jflux[5, i] * (phi[i + nx*ny] - phi[i])
        print(f'old bflux_impl method took {time.time() - t:.4f} seconds')
        # call the function
        t = time.time()
        bflux = _bflux_impl(jflux, ind, self.nx, self.ny, self.nz, phi, full_size=True)
        print(f'_bflux_impl took {time.time() - t:.4f} seconds')
        # check if the result is equal to the reference result
        self.assertTrue(bflux.shape == (6, self.nr))
        self.assertTrue(np.allclose(bflux, ref, rtol=1e-5, atol=1e-8))

def _jflux_impl(nx, ny, nz, r, typ, sigma, eps, eps0, avgeps, with_index=False, 
                full_size=True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    '''kernal implementation of the Jacobian calculation.'''

    # get all indices of the grid points of type "in"
    i = np.where(typ == "in")[0]
    # j stands for flux, <listcomp> faster than for loop
    mydisp_ = [-1, +1, -nx, +nx, -nx*ny, +nx*ny]
    if not full_size:
        jflux = np.array([avgeps(eps[i + d], eps[i]) * eps0 * sigma[i, j // 2] \
                          / np.abs(r[i + d, j // 2] - r[i, j // 2])
                          for j, d in enumerate(mydisp_)]).reshape((6, -1))
        assert jflux.shape == (6, len(i))
    else:
        jflux = np.zeros((6, len(typ)), dtype=float)
        for j, d in enumerate(mydisp_):
            jflux[j, i] =   avgeps(eps[i + d], eps[i]) * eps0 * sigma[i, j // 2] \
                          / np.abs(r[i + d, j // 2] - r[i, j // 2])

    return (jflux, i if with_index else None)

class TestJFluxImplPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e3
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        # set the boundary types
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten()
        self.r = np.random.rand(self.nr, 3).astype(float)
        self.sigma = np.random.rand(self.nr, 3).astype(float)
        self.eps = np.random.rand(self.nr).astype(float)
        self.eps0 = 8.854187817e-12  # vacuum permittivity
        self.avgeps = lambda eps1, eps2: (eps1 + eps2) / 2.0  # arithmetic mean
        
    def test_jflux_impl(self):
        # calculate the reference result
        ref = np.zeros((6, self.nr), dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz
        t = time.time()
        for i in range(self.nr):
            if self.typ[i] == "in":
                ref[0, i] = self.sigma[i, 0] * self.eps0 * self.avgeps(self.eps[i - 1], self.eps[i]) \
                            / np.abs(self.r[i - 1, 0] - self.r[i, 0])
                ref[1, i] = self.sigma[i, 0] * self.eps0 * self.avgeps(self.eps[i + 1], self.eps[i]) \
                            / np.abs(self.r[i + 1, 0] - self.r[i, 0])
                ref[2, i] = self.sigma[i, 1] * self.eps0 * self.avgeps(self.eps[i - nx], self.eps[i]) \
                            / np.abs(self.r[i - nx, 1] - self.r[i, 1])
                ref[3, i] = self.sigma[i, 1] * self.eps0 * self.avgeps(self.eps[i + nx], self.eps[i]) \
                            / np.abs(self.r[i + nx, 1] - self.r[i, 1])
                ref[4, i] = self.sigma[i, 2] * self.eps0 * self.avgeps(self.eps[i - nx*ny], self.eps[i]) \
                            / np.abs(self.r[i - nx*ny, 2] - self.r[i, 2])
                ref[5, i] = self.sigma[i, 2] * self.eps0 * self.avgeps(self.eps[i + nx*ny], self.eps[i]) \
                            / np.abs(self.r[i + nx*ny, 2] - self.r[i, 2])
        print(f'old jflux_impl method took {time.time() - t:.4f} seconds')
        # call the function
        t = time.time()
        jflux, i = _jflux_impl(self.nx, self.ny, self.nz, self.r, self.typ, 
                               self.sigma, self.eps, self.eps0, self.avgeps, 
                               with_index=True, full_size=True)
        print(f'_jflux_impl took {time.time() - t:.4f} seconds')
        # check if the result is equal to the reference result
        self.assertTrue(jflux.shape == (6, self.nr))
        self.assertTrue(i.shape == (len(self.typ[self.typ == "in"]),))
        self.assertTrue(np.allclose(jflux, ref, rtol=1e-5, atol=1e-8))

def _jacobian(inout, flux, i, nx, ny, nz, typ, zfree, phi, phi_, beta):
    '''calculate the Jacobian matrix for the Poisson equation'''
    mydisp_ = [-1, +1, -nx, +nx, -nx*ny, +nx*ny]
    # diagonal elements
    assert flux.shape == (6, len(i))
    inout[i, i]  = -np.sum(flux, axis=0)
    inout[i, i] -= beta * \
        np.abs(coloumb(zfree[i])) * np.exp(-beta * np.sign(zfree[i]) * (phi[i] - phi_[i]))
    # non-diagonal elements
    for jf, d in zip(flux, mydisp_):
        inout[i, i + d] = jf
    # impose the boundary conditions
    _impose_j_bound(inout, nx, ny, nz, typ, val=1.0, mask=-1.0)

def _rhsvec(inout, flux, i, nx, ny, nz, typ, zfree, zfixed, phi, phi_, dirichlet_pot, beta):
    '''calculate the right-hand side vector for the Poisson equation'''
    # calculate the b vector
    assert flux.shape == (6, len(i))
    inout[i]  = np.sum(flux, axis=0)
    inout[i] += coloumb(zfree[i]) * \
        np.exp(-beta * np.sign(zfree[i]) * (phi[i] - phi_[i]))
    inout[i] += coloumb(zfixed[i])
    # impose the boundary conditions
    _impose_b_bound(inout, nx, ny, nz, typ, phi, dirichlet_pot)
    
    inout *= -1 # for convenience change the sign of B in later NR iteration

def calculate(jinout: Optional[lil_matrix], 
              binout: Optional[np.ndarray],
              grid_dim: Tuple[int, int, int]|List[int],
              gridpoint_coords: np.ndarray, 
              gridpoint_typ: List[str]|np.ndarray,
              gridpoint_surfarea: np.ndarray, 
              eps: np.ndarray, 
              phi: np.ndarray, 
              phi_: np.ndarray, 
              free_chr: np.ndarray,
              fixed_chr: Optional[np.ndarray] = None,
              dirichlet_pot: Optional[np.ndarray] = None,
              eps0: float = 8.854187817e-12, 
              beta: float = 1/(1.380649e-23 * 300), 
              feps: Callable[[float, float], float] = lambda eps1, eps2: (eps1 + eps2) / 2.0) -> None:
    '''
    refactored version of the implementation to calculate the Jacobian for the Poisson equation
    using the Newton-Raphson method (finit difference method).
    
    NOTE: the input would be modified in-place
    
    Parameters
    ----------
    jinout : scipy.sparse.lil_matrix
        The Jacobian matrix to be calculated. It is a sparse matrix in LIL format. If is None,
        then the Jacobian will not be calculated.
    binout : np.ndarray
        The b vector to be calculated, which is the right-hand side of the linear system.
        It is a dense numpy array. If is None, then the b vector will not be calculated.
    grid_dim : tuple or list of int
        The dimensions of the grid as (nx, ny, nz), where nx, ny, and nz are the number of grid 
        points in the x, y, and z directions, respectively.
    gridpoint_coords : np.ndarray
        The coordinates of the grid points, shape (nr, 3), where nr is the total number of grid
        points.
    gridpoint_typ : list of str
        The type of grid point, with length nr
    gridpoint_surfarea : np.ndarray
        The area of Voronoi surface in three dimension, shape (nr, 3). For example the gridpoint_
        area[ir, 0] indicates the area of surface whose norm vector is along x-axis
    eps : np.ndarray
        The relative permittivity of each grid point
    phi : np.ndarray
        The electrostatic potential mapped on real-space grid point
    phi\_ : np.ndarray
        The "old" electrostatic potential mapped on real-space grid point
    free_chr : np.ndarray
        The free charge density mapped on real-space grid point
    fixed_chr : np.ndarray, optional
        The fixed charge density mapped on real-space grid point, shape (nr,). 
    dirichlet_pot : np.ndarray, optional
        The Dirichlet potential mapped on real-space grid point, shape (nr,). 
    eps0 : float, optional
        The vacuum permittivity, a constant value. Default is 8.854187817e-12 F/m.
    beta : float, optional
        The inverse temperature (1/kBT), used in the Boltzmann factor for charge density.
        Default is 1/(1.380649e-23 * 300), which corresponds to room temperature (300 K). 
    feps : callable, optional
        the functor to calculate the average permittivity on a pair of grid points.
        It should take two arguments, eps1 and eps2, and return the average permittivity.
        If not provided, it defaults to the arithmetic mean.
    '''
    # shortcut: nothing to do
    if jinout is None and binout is None:
        return
    
    # check dimensions
    if not isinstance(grid_dim, (tuple, list)) or len(grid_dim) != 3:
        raise ValueError("grid_dim must be a tuple or list of three integers")
    if any(not isinstance(d, int) or d <= 0 for d in grid_dim):
        raise ValueError("grid_dim must contain positive integers")
    nx, ny, nz = grid_dim; nr = nx * ny * nz
    
    # simple sanity check
    if jinout is not None:
        if not isinstance(jinout, lil_matrix):
            raise TypeError("jinout (jacobian) must be a scipy.sparse.lil_matrix")
        if jinout.shape != (nr, nr):
            raise ValueError(f"jinout (jacobian) must have shape ({nr}, {nr}), but got {jinout.shape}")

    if binout is not None:
        if not isinstance(binout, np.ndarray):
            raise TypeError("binout must be a numpy array")
        if binout.shape != (nr,):
            raise ValueError(f"binout must have shape ({nr},), but got {binout.shape}")
    
    # check the common input parameters
    if not isinstance(gridpoint_coords, np.ndarray) or gridpoint_coords.shape != (nr, 3):
        raise ValueError(f"gridpoint_coords must be a numpy array of shape ({nr}, 3)")
    gridpoint_typ = np.array(list(gridpoint_typ.values()), dtype=str) \
        if isinstance(gridpoint_typ, dict) else np.array(gridpoint_typ, dtype=str)
    if not isinstance(gridpoint_typ, np.ndarray) or len(gridpoint_typ) != nr:
        raise ValueError(f"gridpoint_typ must be a list of length {nr}")
    if not isinstance(gridpoint_surfarea, np.ndarray) or gridpoint_surfarea.shape != (nr, 3):
        raise ValueError(f"gridpoint_surfarea must be a numpy array of shape ({nr}, 3)")
    if not isinstance(eps, np.ndarray) or eps.shape != (nr,):
        raise ValueError(f"eps must be a numpy array of shape ({nr},)")
    if not isinstance(phi, np.ndarray) or phi.shape != (nr,):
        raise ValueError(f"phi must be a numpy array of shape ({nr},)")
    if not isinstance(phi_, np.ndarray) or phi_.shape != (nr,):
        raise ValueError(f"phi_ must be a numpy array of shape ({nr},)")
    if not isinstance(free_chr, np.ndarray) or free_chr.shape != (nr,):
        raise ValueError(f"free_chr must be a numpy array of shape ({nr},)")
    
    if binout is not None:
        if not isinstance(fixed_chr, np.ndarray) or fixed_chr.shape != (nr,):
            raise ValueError(f"fixed_chr must be a numpy array of shape ({nr},)")
        if not isinstance(dirichlet_pot, np.ndarray) or dirichlet_pot.shape != (nr,):
            raise ValueError(f"dirichlet_pot must be a numpy array of shape ({nr},)")

    if not isinstance(eps0, (float, int)):
        raise TypeError("eps0 must be a float or int")
    if not isinstance(beta, (float, int)):
        raise TypeError("beta must be a float or int")
    if not callable(feps):
        raise TypeError("feps must be a callable function with strictly two arguments: "
                        "eps1 and eps2 to calculate the average permittivity")
    
    # ================================ implementation =======================================
    # making alias for more succint code
    r      = gridpoint_coords
    typ    = np.array(gridpoint_typ, dtype=str)
    sigma  = gridpoint_surfarea
    zfree  = free_chr
    zfixed = fixed_chr
    
    # calculate the flux of jacobian for all grid points of type "in"
    jflux, ind = _jflux_impl(nx, ny, nz, r, typ, sigma, eps, eps0, feps, 
                             with_index=True, full_size=False)
    # the "ind" is the indices of the grid points of type "in"
    assert jflux.shape == (6, len(ind)) # ensure the full_size is disabled
    
    if jinout is not None: # only calculate jacobian when jinout is provided (allocated)
        _jacobian(jinout, jflux, ind, nx, ny, nz, typ, zfree, phi, phi_, beta)
        assert jinout.shape == (nr, nr)

    if binout is not None: # only calculate b vector when binout is provided (allocated)
        # calculate bflux with jflux
        bflux = _bflux_impl(jflux, ind, nx, ny, nz, phi, full_size=False)
        assert bflux.shape == (6, len(ind))
        _rhsvec(binout, bflux, ind, nx, ny, nz, typ, zfree, zfixed, phi, phi_, 
                dirichlet_pot, beta)
        assert binout.shape == (nr,)

if __name__ == '__main__':
    unittest.main()
