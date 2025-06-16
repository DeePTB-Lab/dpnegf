import numpy as np
import pytest
from dpnegf.negf.scf_method import AndersonMixer
from dpnegf.negf.scf_method import BroydenSecondMixer

def test_anderson_mixer_linear_mixing_behavior():
    mixer = AndersonMixer(m=3, alpha=0.5, num_linear_warmup=2)
    x0 = np.array([1.0, 2.0])
    f0 = np.array([2.0, 4.0])
    # First update: should use linear mixing
    x1 = mixer.update(f0, x0)
    expected_x1 = x0 + 0.5 * (f0 - x0)
    np.testing.assert_allclose(x1, expected_x1)
    # Second update: still linear mixing
    f1 = np.array([3.0, 6.0])
    x2 = mixer.update(f1, x1)
    expected_x2 = x1 + 0.5 * (f1 - x1)
    np.testing.assert_allclose(x2, expected_x2)

def test_anderson_mixer_switches_to_anderson():
    mixer = AndersonMixer(m=2, alpha=0.1, num_linear_warmup=1)
    x0 = np.array([0.0, 0.0])
    f0 = np.array([1.0, 1.0])
    # First update: linear mixing
    x1 = mixer.update(f0, x0)
    np.testing.assert_allclose(x1, x0 + 0.1 * (f0 - x0))
    # Second update: should switch to Anderson mixing
    f1 = np.array([0.6, 0.4])
    x2 = mixer.update(f1, x1)
    # Anderson mixing should not raise and should return a numpy array of correct shape
    assert isinstance(x2, np.ndarray)
    assert x2.shape == x1.shape
    x2_ = np.array([0.35135135, 0.02702703])
    assert  abs(x2 - x2_).max() < 1e-8

def test_anderson_mixer_reset():
    mixer = AndersonMixer(m=2, alpha=0.3, num_linear_warmup=1)
    x0 = np.array([1.0, 1.0])
    f0 = np.array([2.0, 2.0])
    mixer.update(f0, x0)
    mixer.reset()
    assert mixer.iter == 0
    assert mixer.xkm1 is None
    assert mixer.fkm1 is None
    assert mixer.dx_hist == []
    assert mixer.df_hist == []
    assert mixer.first_linear is True

def test_anderson_mixer_shape_assertion():
    mixer = AndersonMixer()
    x = np.array([1.0, 2.0])
    f = np.array([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError):
        mixer.update(f, x)

def test_broyden_second_mixer_linear_first_step():
    # Test that the first update is equivalent to linear mixing with -alpha
    shape = (2,)
    alpha = 0.3
    mixer = BroydenSecondMixer(shape, alpha=alpha)
    x0 = np.array([1.0, 2.0])
    f0 = np.array([0.5, -1.0])
    x1 = mixer.update(x0, f0)
    expected = x0 + alpha * f0
    np.testing.assert_allclose(x1, expected)

def test_broyden_second_mixer_second_step_update():
    # Test that the second update uses the Broyden formula and returns correct shape
    shape = (2,)
    alpha = 0.2
    mixer = BroydenSecondMixer(shape, alpha=alpha)
    x0 = np.array([0.0, 0.0])
    f0 = np.array([1.0, -1.0])
    x1 = mixer.update(x0, f0)
    x2 = mixer.update(x1, np.array([0.5, -0.5]))
    x2_ = np.array([0.4, -0.4])  # Expected value after second update
    assert isinstance(x2, np.ndarray)
    assert x2.shape == x0.shape
    np.testing.assert_allclose(x2, x2_)

def test_broyden_second_mixer_reset():
    shape = (3,)
    mixer = BroydenSecondMixer(shape, alpha=0.1)
    x0 = np.array([1.0, 2.0, 3.0])
    f0 = np.array([0.1, 0.2, 0.3])
    mixer.update(x0, f0)
    mixer.reset(shape)
    assert mixer.iter == 0
    np.testing.assert_array_equal(mixer.x_last, np.zeros(shape))
    np.testing.assert_array_equal(mixer.f_last, np.zeros(shape))
    assert isinstance(mixer.H0, np.ndarray)
    assert mixer.df_hist == []

def test_broyden_second_mixer_multiple_iterations():
    # Test that the mixer can run for several iterations without error
    shape = (2,)
    mixer = BroydenSecondMixer(shape, alpha=0.15)
    x = np.array([0.0, 0.0])
    xlist_ = np.array([[0.15,-0.15],
                       [1.5,-1.5],
                       [12.3,-12.3],
                       [87.8999999,-87.8999999],
                       [541.4999999,-541.4999999]])
    for i in range(5):
        f = np.array([1.0 - 0.1 * i, -1.0 + 0.1 * i])
        x = mixer.update(x, f)
        assert x.shape == (2,)
        assert np.allclose(x, xlist_[i])

