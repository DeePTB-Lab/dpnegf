import re
from typing import Optional

import torch
import numpy as np
import ase
from dptb.data import AtomicData
from dpnegf.negf.poisson_init import Grid, Interface3D, Dirichlet, Dielectric

class PoissonEquation:
    '''
    this class defines the pure mathematical problem (poisson equation)
    that can only be operated with mathematical operations
    '''
    @staticmethod
    def build_dirichlet_boundary_conditions(**kwargs):
        '''
        build the Dirichlet boundary conditions for the Poisson equation.
        Consider there may be multiple Dirichlet boundary conditions, 
        e.g., multiple gates or leads, all must be provided in a dictionary
        with keys starting with "gate" or "lead".
        
        Parameters
        ----------
        kwargs : dict
            the keyword arguments for the Dirichlet boundary conditions,
            e.g., {'gate1': {'x_range': '0:10:100',
                             'y_range': '0:10:100',
                             'z_range': '0:10:100',
                             'voltage': 1.0},
                    'lead_L': {'x_range': '0:10:100',
                               'y_range': '0:10:100',
                               'z_range': '0:10:100',
                               'voltage': 0.0}}
        
        Returns
        -------
        list
            a list of Dirichlet boundary conditions, each is an instance of Dirichlet class
        '''
        mybc = {k: Dirichlet(x_range=v.get('x_range').split(':'),
                             y_range=v.get('y_range').split(':'),
                             z_range=v.get('z_range').split(':'))
                for k, v in kwargs.items() if re.match(r'^gate|lead', k)}
        if len(mybc) == 0:
            raise ValueError('Failed to build Dirichlet boundary conditions. '
                             'Please check if the input parameters have keys '
                             'starting with "gate" or "lead".')
        
        for k, bc in mybc.items():
            bc.Ef = -1*float(kwargs[k].get('voltage', 0.0))
        return list(mybc.values())

    @staticmethod
    def generate_real_space_grid(xrange : str, 
                                 yrange : str, 
                                 zrange : str, 
                                 coords: np.ndarray):
        '''
        generate the evenly spaced real space grid for the Poisson equation
        '''
        xlo, xhi, nx = map(float, xrange.split(':'))
        xg = np.linspace(xlo, xhi, int(nx))
        
        ylo, yhi, ny = map(float, yrange.split(':'))
        yg = np.linspace(ylo, yhi, int(ny))
        # TODO: temporary fix for 2D cases
        # yg = np.array((ylo+yhi)/2)
        
        zlo, zhi, nz = map(float, zrange.split(':'))
        zg = np.linspace(zlo, zhi, int(nz))
        
        # auto-unpack the positions of atoms
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
            # this may raise error when the conversion is not possible
        assert coords.shape[1] == 3, "Coordinates should have shape (N, 3)"
        xa, ya, za = coords.T # unpack the (3, N)
        
        # return Grid(xg, yg, zg, xa, ya, za) # TODO: change back to zg
        return Grid(xg, yg, za, xa, ya, za)
    
    def __init__(self,
                 real_space_grid: Grid,
                 rho,
                 bc,
                 bc_type: Optional[str] = 'Dirichlet'):
        self.bc = bc
        self._impl = Interface3D(real_space_grid, bc, None)
    
    def initialize_grid_points(self):
        self._impl.get_potential_eps(self.bc)
    
    def solve(self,
              method : str,
              tol : float,
              dtype,
              maxiter : int = 1000,):
        '''
        solve the Poisson equation with the given boundary conditions
        
        Returns
        -------
        potential : any
            the potential solution of the Poisson equation
        '''
        raise NotImplementedError("The solve method is not implemented yet for "
                                  "non-interfaced Poisson equation. ")

class InterfacedPoissonEquation(PoissonEquation):
    '''
    this class defines the Poisson equation with interface
    '''
    @staticmethod
    def define_dielectric_regions(**kwargs):
        '''
        build the dielectric boundary conditions for the Poisson equation
        '''
        mybc = {k: Dielectric(x_range=v.get('x_range').split(':'),
                              y_range=v.get('y_range').split(':'),
                              z_range=v.get('z_range').split(':'))
                for k, v in kwargs.items() if re.match(r'^dielectric', k)}
        if len(mybc) == 0:
            raise ValueError('Failed to build dielectric boundary conditions. '
                             'Please check if the input parameters have keys '
                             'starting with "gate" or "lead".')
        for k, bc in mybc.items():
            bc.eps = -1*float(kwargs[k].get('relative permittivity', 0.0))
        return list(mybc.values())
    
    def __init__(self,
                 real_space_grid: Grid, 
                 rho, 
                 bc, 
                 bc_type = 'Dirichlet',
                 dielectric = None):
        '''
        define the Poisson equation with interface
        '''
        assert dielectric is not None, "Dielectric regions must be provided."
        self.bc = bc
        self.dielectric = dielectric
        self._impl = Interface3D(real_space_grid, self.bc, self.dielectric)
        self.atom_grid_point_indexes = None
        
    def initialize_grid_points(self, spatial_charge_regions=None):
        self._impl.get_potential_eps(self.bc + self.dielectric)
        if spatial_charge_regions is not None:
            self.atom_grid_point_indexes = list(self._impl.grid.atom_index_dict.values())
            for r in spatial_charge_regions:
                self._impl.get_fixed_charge(r.get('x_range').split(':'),
                                            r.get('y_range').split(':'),
                                            r.get('z_range').split(':'),
                                            r.get('charge'),
                                            self.atom_grid_point_indexes)

    def solve(self,
              method : str = 'pyamg',
              tol : float = 1e-7,
              dtype: str = 'float64',
              maxiter : int = 1000):
        '''
        solve the Poisson equation with the given boundary conditions
        
        Returns
        -------
        potential : any
            the potential solution of the Poisson equation
        '''
        return self._impl.solve_poisson_NRcycle(method=method, 
                                                tolerance=tol, 
                                                dtype=dtype)

class NonEquilibriumGreenFunction:
    '''
    this class defines the pure mathematical problem (NEGF equations)
    that can only be operated with mathematical operations
    '''
    def __init__(self):
        '''
        is it possible to make the NEGF equations individual without any
        dependency on the structure?
        '''
        pass
    
    def solve(self):
        '''
        solve the NEGF equations
        '''
        pass