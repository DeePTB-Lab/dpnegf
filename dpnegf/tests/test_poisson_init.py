import numpy as np
import pytest
from dpnegf.negf.poisson_init import Grid
from dpnegf.negf.poisson_init import region
from dpnegf.negf.poisson_init import Dirichlet
from dpnegf.negf.poisson_init import Dielectric

def test_grid_basic_properties():
    # Define a simple 2x2x2 grid and 2 atoms inside the grid
    xg = np.array([0.0, 1.0])
    yg = np.array([0.0, 1.0])
    zg = np.array([0.0, 1.0])
    xa = np.array([0.0, 1.0])
    ya = np.array([0.0, 1.0])
    za = np.array([0.0, 1.0])

    grid = Grid(xg, yg, zg, xa, ya, za)

    # Check grid properties
    assert np.allclose(grid.xg, xg)
    assert np.allclose(grid.yg, yg)
    assert np.allclose(grid.zg, zg)
    assert grid.Na == 2
    assert grid.shape == (2, 2, 2)
    assert grid.Np == 8
    assert grid.grid_coord.shape == (8, 3)
    assert isinstance(grid.atom_index_dict, dict)
    assert set(grid.atom_index_dict.keys()) == {0, 1}
    assert grid.surface_grid.shape == (8, 3)
    
    grid_surface_grid_std= [[0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25]]
    assert np.allclose(grid.surface_grid, grid_surface_grid_std)


def test_grid_atom_index():
    # Atoms at grid points
    xg = np.array([0.0, 1.0, 2.0])
    yg = np.array([0.0, 1.0])
    zg = np.array([0.0, 1.0])
    xa = np.array([1.0, 2.0])
    ya = np.array([1.0, 0.0])
    za = np.array([0.0, 1.0])

    grid = Grid(xg, yg, zg, xa, ya, za)
    # Check that atom_index_dict maps each atom to a valid grid point
    for atom_idx, grid_idx in grid.atom_index_dict.items():
        atom_pos = (xa[atom_idx], ya[atom_idx], za[atom_idx])
        grid_pos = grid.grid_coord[grid_idx]
        assert np.allclose(atom_pos, grid_pos, atol=1e-3)

def test_grid_cal_vorlen_uniform():
    # Uniform grid
    x = np.array([0.0, 1.0, 2.0, 3.0])
    grid = Grid(x, x, x, x, x, x)
    vorlen = grid.cal_vorlen(x)
    # Endpoints: half the distance to neighbor, middle: average of neighbors
    assert np.isclose(vorlen[0], 0.5)
    assert np.isclose(vorlen[-1], 0.5)
    assert np.allclose(vorlen[1:-1], 1.0)

def test_grid_cal_vorlen_nonuniform():
    # Non-uniform grid
    x = np.array([0.0, 1.0, 3.0, 6.0])
    grid = Grid(x, x, x, x, x, x)
    vorlen = grid.cal_vorlen(x)
    assert np.isclose(vorlen[0], 0.5)
    assert np.isclose(vorlen[1], (1.0 + 2.0) / 2)
    assert np.isclose(vorlen[2], (2.0 + 3.0) / 2)
    assert np.isclose(vorlen[3], 1.5)

def test_grid_atom_outside_raises():
    # Atom outside grid should raise AssertionError
    xg = np.array([0.0, 1.0])
    yg = np.array([0.0, 1.0])
    zg = np.array([0.0, 1.0])
    xa = np.array([-1.0])  # outside
    ya = np.array([0.5])
    za = np.array([0.5])
    with pytest.raises(AssertionError):
        Grid(xg, yg, zg, xa, ya, za)

def test_region_attributes():
    x_range = (0, 10)
    y_range = (1, 5)
    z_range = (-2, 2)
    r = region(x_range, y_range, z_range)
    assert r.xmin == 0.0
    assert r.xmax == 10.0
    assert r.ymin == 1.0
    assert r.ymax == 5.0
    assert r.zmin == -2.0
    assert r.zmax == 2.0

def test_region_float_input():
    x_range = (0.5, 2.5)
    y_range = (1.1, 3.3)
    z_range = (4.4, 5.5)
    r = region(x_range, y_range, z_range)
    assert r.xmin == 0.5
    assert r.xmax == 2.5
    assert r.ymin == 1.1
    assert r.ymax == 3.3
    assert r.zmin == 4.4
    assert r.zmax == 5.5

def test_region_list_input():
    x_range = [1, 2]
    y_range = [3, 4]
    z_range = [5, 6]
    r = region(x_range, y_range, z_range)
    assert r.xmin == 1.0
    assert r.xmax == 2.0
    assert r.ymin == 3.0
    assert r.ymax == 4.0
    assert r.zmin == 5.0
    assert r.zmax == 6.0

def test_dirichlet_inherits_region():
    x_range = (0, 1)
    y_range = (2, 3)
    z_range = (4, 5)
    d = Dirichlet(x_range, y_range, z_range)
    # Check region attributes
    assert d.xmin == 0.0
    assert d.xmax == 1.0
    assert d.ymin == 2.0
    assert d.ymax == 3.0
    assert d.zmin == 4.0
    assert d.zmax == 5.0

def test_dirichlet_default_ef():
    d = Dirichlet((0, 1), (0, 1), (0, 1))
    assert hasattr(d, "Ef")
    assert d.Ef == 0.0

def test_dirichlet_accepts_list_input():
    d = Dirichlet([1, 2], [3, 4], [5, 6])
    assert d.xmin == 1.0
    assert d.xmax == 2.0
    assert d.ymin == 3.0
    assert d.ymax == 4.0
    assert d.zmin == 5.0
    assert d.zmax == 6.0

def test_dielectric_inherits_region():
    x_range = (0, 2)
    y_range = (1, 3)
    z_range = (4, 5)
    d = Dielectric(x_range, y_range, z_range)
    assert d.xmin == 0.0
    assert d.xmax == 2.0
    assert d.ymin == 1.0
    assert d.ymax == 3.0
    assert d.zmin == 4.0
    assert d.zmax == 5.0

def test_dielectric_default_eps():
    d = Dielectric((0, 1), (0, 1), (0, 1))
    assert hasattr(d, "eps")
    assert d.eps == 1.0

def test_dielectric_accepts_list_input():
    d = Dielectric([1, 2], [3, 4], [5, 6])
    assert d.xmin == 1.0
    assert d.xmax == 2.0
    assert d.ymin == 3.0
    assert d.ymax == 4.0
    assert d.zmin == 5.0
    assert d.zmax == 6.0




