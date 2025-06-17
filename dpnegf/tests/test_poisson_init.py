import numpy as np
import pytest
from dpnegf.negf.poisson_init import Grid
from dpnegf.negf.poisson_init import region
from dpnegf.negf.poisson_init import Dirichlet
from dpnegf.negf.poisson_init import Dielectric
from dpnegf.negf.poisson_init import Grid, region, Dirichlet, Dielectric, Interface3D
import sys
from scipy.sparse import lil_matrix

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


class DummyLog:
    def info(self, msg): pass
    def warning(self, msg): pass

# Patch log and constants for Interface3D
sys.modules['dpnegf.negf.poisson_init'].log = DummyLog()
sys.modules['dpnegf.negf.poisson_init'].Boltzmann = 1.380649e-23
sys.modules['dpnegf.negf.poisson_init'].eV2J = 1.602176634e-19
sys.modules['dpnegf.negf.poisson_init'].eps0 = 8.854187817e-12
sys.modules['dpnegf.negf.poisson_init'].elementary_charge = 1.602176634e-19
from scipy.sparse import csr_matrix
sys.modules['dpnegf.negf.poisson_init'].csr_matrix = lambda *a, **k: csr_matrix(*a, **k)
sys.modules['dpnegf.negf.poisson_init'].spsolve = lambda A, b: np.linalg.solve(A.toarray(), b) if hasattr(A, 'toarray') else np.linalg.solve(A, b)

def make_simple_grid():
    xg = np.array([0.0, 1.0])
    yg = np.array([0.0, 1.0])
    zg = np.array([0.0, 1.0])
    xa = np.array([0.0, 1.0])
    ya = np.array([0.0, 1.0])
    za = np.array([0.0, 1.0])
    return Grid(xg, yg, zg, xa, ya, za)

def test_interface3d_init_and_boundary_points():
    grid = make_simple_grid()
    d = Dirichlet((0, 0), (0, 1), (0, 1))
    diel = Dielectric((0, 1), (0, 1), (0, 1))
    iface = Interface3D(grid, [d], [diel])
    # Check attributes
    assert iface.grid is grid
    assert iface.Dirichlet_group == [d]
    assert iface.dielectric_group == [diel]
    assert iface.eps.shape == (grid.Np,)
    assert iface.phi.shape == (grid.Np,)
    assert iface.free_charge.shape == (grid.Np,)
    assert iface.fixed_charge.shape == (grid.Np,)
    assert iface.lead_gate_potential.shape == (grid.Np,)
    # Check boundary points
    boundary_types = set(iface.boudnary_points.values())
    assert "in" not in boundary_types
    assert hasattr(iface, "internal_NP")
    assert isinstance(iface.internal_NP, int)
    boundary_std = {0: 'xmin', 1: 'xmax', 2: 'xmin', 3: 'xmax', 4: 'xmin', 5: 'xmax', 6: 'xmin', 7: 'xmax'}
    assert iface.boudnary_points == boundary_std
    assert iface.internal_NP == 0  # No internal points in this simple grid

def test_interface3d_get_fixed_charge_sets_values():
    grid = make_simple_grid()
    d = Dirichlet((0, 0), (0, 1), (0, 1))
    diel = Dielectric((0, 1), (0, 1), (0, 1))
    iface = Interface3D(grid, [d], [diel])
    # All atoms are at grid points 0 and 7
    atom_indices = np.array(list(grid.atom_index_dict.values()))
    iface.get_fixed_charge((0, 1), (0, 1), (0, 1), 0.5, atom_indices)
    # Only grid points corresponding to atoms should be set
    for i in range(grid.Np):
        if i in atom_indices:
            assert iface.fixed_charge[i] == 0.5
        else:
            assert iface.fixed_charge[i] == 0.0

def test_interface3d_get_potential_eps_dirichlet_and_dielectric():
    grid = make_simple_grid()
    d = Dirichlet((0, 0), (0, 1), (0, 1))
    d.Ef =2.0
    diel = Dielectric((0, 1), (0, 1), (0, 1))
    diel.eps = 5.0
    iface = Interface3D(grid, [d], [diel])
    iface.get_potential_eps([d, diel])
    # Dirichlet region should set lead_gate_potential to -Ef at correct indices
    idx = np.nonzero((d.xmin <= grid.grid_coord[:,0]) &
                        (d.xmax >= grid.grid_coord[:,0]) &
                        (d.ymin <= grid.grid_coord[:,1]) &
                        (d.ymax >= grid.grid_coord[:,1]) &
                        (d.zmin <= grid.grid_coord[:,2]) &
                        (d.zmax >= grid.grid_coord[:,2]))[0]
    assert np.allclose(iface.lead_gate_potential[idx], -2.0)
    # Dielectric region should set eps to 5.0 everywhere
    assert np.allclose(iface.eps, 5.0)

def test_interface3d_get_potential_eps_raises_on_unknown_type():
    grid = make_simple_grid()
    class Dummy: xmin=0; xmax=1; ymin=0; ymax=1; zmin=0; zmax=1
    iface = Interface3D(grid, [], [])
    with pytest.raises(ValueError):
        iface.get_potential_eps([Dummy()])

def test_interface3d_to_scipy_Jac_B_shapes():
    grid = make_simple_grid()
    d = Dirichlet((0, 0), (0, 1), (0, 1))
    diel = Dielectric((0, 1), (0, 1), (0, 1))
    iface = Interface3D(grid, [d], [diel])
    J, B = iface.to_scipy_Jac_B()
    assert isinstance(J, csr_matrix)
    print(f"J shape: {J.shape}")
    print("J:", J.toarray())
    J_std = np.array(   [[ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  1., -1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0., -1.,  1.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  1., -1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  1.],])
    assert J.shape == (grid.Np, grid.Np)
    assert J.toarray().shape == (grid.Np, grid.Np)
    assert np.allclose(J.toarray(), J_std)
    B_std = np.array([0, 0, 0. ,0. ,0. ,0. ,0. ,0.])
    assert B.shape == (grid.Np,)
    assert np.allclose(B, B_std)

def test_interface3d_to_pyamg_Jac_B_shapes():
    grid = make_simple_grid()
    d = Dirichlet((0, 0), (0, 1), (0, 1))
    diel = Dielectric((0, 1), (0, 1), (0, 1))
    iface = Interface3D(grid, [d], [diel])
    J, B = iface.to_pyamg_Jac_B()
    J_std = np.array([  [ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  1., -1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
                        [ 0.,  0.,  0.,  0., -1.,  1.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  1., -1.],
                        [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  1.],])
    B_std = np.array([0, 0, 0. ,0. ,0. ,0. ,0. ,0.])
    assert isinstance(J, csr_matrix)
    assert J.toarray().shape == (grid.Np, grid.Np)
    assert np.allclose(J.toarray(), J_std)
    assert isinstance(B, np.ndarray)
    assert np.allclose(B, B_std)
    assert J.shape == (grid.Np, grid.Np)
    assert B.shape == (grid.Np,)

def test_interface3d_solve_poisson_NRcycle_dtype_check():
    grid = make_simple_grid()
    d = Dirichlet((0, 0), (0, 1), (0, 1))
    diel = Dielectric((0, 1), (0, 1), (0, 1))
    iface = Interface3D(grid, [d], [diel])
    with pytest.raises(ValueError):
        iface.solve_poisson_NRcycle(dtype="unknown")

def test_interface3d_solve_poisson_NRcycle_method_check():
    grid = make_simple_grid()
    d = Dirichlet((0, 0), (0, 1), (0, 1))
    diel = Dielectric((0, 1), (0, 1), (0, 1))
    iface = Interface3D(grid, [d], [diel])
    with pytest.raises(ValueError):
        iface.solve_poisson_NRcycle(method="unknown")

def test_interface3d_NR_construct_Jac_B_boundary_and_internal():
    grid = make_simple_grid()
    d = Dirichlet((0, 0), (0, 1), (0, 1))
    diel = Dielectric((0, 1), (0, 1), (0, 1))
    iface = Interface3D(grid, [d], [diel])
    J = lil_matrix((grid.Np, grid.Np))
    B = np.zeros(grid.Np)
    iface.NR_construct_Jac_B(J, B)
    # Check that diagonal is set for all points
    diag = J.diagonal()
    assert np.all(diag == 1)
    J_std = np.array([[ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                [ 0.,  0.,  1., -1.,  0.,  0.,  0.,  0.],
                [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
                [ 0.,  0.,  0.,  0., -1.,  1.,  0.,  0.],
                [ 0.,  0.,  0.,  0.,  0.,  0.,  1., -1.],
                [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  1.],])
    assert np.allclose(J.toarray(), J_std)
    # Check that B is a vector of correct shape
    assert B.shape == (grid.Np,)
    B_std = np.array([0., 0., 0., 0., 0. ,0. ,0., 0.])
    assert np.allclose(B, B_std)





