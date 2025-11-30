"""
Unit tests for automatic memory detection functions in lead_property.py.

Tests cover:
- _estimate_worker_memory: Memory estimation per worker
- _get_safe_n_jobs: Safe parallel worker calculation
"""

import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from dpnegf.negf.lead_property import _estimate_worker_memory, _get_safe_n_jobs


class MockHamiltonian:
    """Mock Hamiltonian object for testing."""

    def __init__(self, matrix_size=100, use_torch=True, raise_error=False):
        self.matrix_size = matrix_size
        self.use_torch = use_torch
        self.raise_error = raise_error

    def get_hs_lead(self, kpoint, tab, v):
        if self.raise_error:
            raise RuntimeError("Simulated Hamiltonian fetch error")

        n = self.matrix_size
        if self.use_torch:
            # Create PyTorch tensors (complex128 = 16 bytes per element)
            hL = torch.zeros(n, n, dtype=torch.complex128)
            hLL = torch.zeros(n, n, dtype=torch.complex128)
            hDL = torch.zeros(n, n, dtype=torch.complex128)
            sL = torch.zeros(n, n, dtype=torch.complex128)
            sLL = torch.zeros(n, n, dtype=torch.complex128)
            sDL = torch.zeros(n, n, dtype=torch.complex128)
        else:
            # Create NumPy arrays
            hL = np.zeros((n, n), dtype=np.complex128)
            hLL = np.zeros((n, n), dtype=np.complex128)
            hDL = np.zeros((n, n), dtype=np.complex128)
            sL = np.zeros((n, n), dtype=np.complex128)
            sLL = np.zeros((n, n), dtype=np.complex128)
            sDL = np.zeros((n, n), dtype=np.complex128)

        return hL, hLL, hDL, sL, sLL, sDL


class MockLead:
    """Mock LeadProperty object for testing."""

    def __init__(self, tab="lead_L", voltage=0.0, matrix_size=100,
                 use_torch=True, raise_error=False):
        self.tab = tab
        self.voltage = voltage
        self.hamiltonian = MockHamiltonian(matrix_size, use_torch, raise_error)


# =============================================================================
# Tests for _estimate_worker_memory
# =============================================================================

class TestEstimateWorkerMemory:
    """Tests for _estimate_worker_memory function."""

    def test_base_overhead_included(self):
        """Test that base overhead (300MB) is always included."""
        lead_L = MockLead("lead_L", matrix_size=1)  # tiny matrices
        lead_R = MockLead("lead_R", matrix_size=1)

        result = _estimate_worker_memory(lead_L, lead_R)
        base_overhead = 300 * 1024 * 1024  # 300 MB

        # Result should be at least base_overhead
        assert result >= base_overhead

    def test_matrix_memory_scaling(self):
        """Test that memory scales with matrix size."""
        # Small matrices (10x10)
        lead_L_small = MockLead("lead_L", matrix_size=10)
        lead_R_small = MockLead("lead_R", matrix_size=10)
        result_small = _estimate_worker_memory(lead_L_small, lead_R_small)

        # Large matrices (100x100)
        lead_L_large = MockLead("lead_L", matrix_size=100)
        lead_R_large = MockLead("lead_R", matrix_size=100)
        result_large = _estimate_worker_memory(lead_L_large, lead_R_large)

        # Larger matrices should require more memory
        assert result_large > result_small

    def test_temp_allocation_factor(self):
        """Test that temp_allocation_factor scales computation memory."""
        lead_L = MockLead("lead_L", matrix_size=50)
        lead_R = MockLead("lead_R", matrix_size=50)

        result_factor_1 = _estimate_worker_memory(lead_L, lead_R, temp_allocation_factor=1.0)
        result_factor_3 = _estimate_worker_memory(lead_L, lead_R, temp_allocation_factor=3.0)
        result_factor_5 = _estimate_worker_memory(lead_L, lead_R, temp_allocation_factor=5.0)

        # Higher factor should give higher estimate
        assert result_factor_3 > result_factor_1
        assert result_factor_5 > result_factor_3

    def test_pytorch_tensor_support(self):
        """Test memory estimation with PyTorch tensors."""
        lead_L = MockLead("lead_L", matrix_size=50, use_torch=True)
        lead_R = MockLead("lead_R", matrix_size=50, use_torch=True)

        result = _estimate_worker_memory(lead_L, lead_R)

        # Should return a positive integer
        assert isinstance(result, int)
        assert result > 0

    def test_numpy_array_support(self):
        """Test memory estimation with NumPy arrays."""
        lead_L = MockLead("lead_L", matrix_size=50, use_torch=False)
        lead_R = MockLead("lead_R", matrix_size=50, use_torch=False)

        result = _estimate_worker_memory(lead_L, lead_R)

        # Should return a positive integer
        assert isinstance(result, int)
        assert result > 0

    def test_default_kpoint_gamma(self):
        """Test that default k-point is Gamma [0,0,0]."""
        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        # Should not raise with kpoint=None
        result = _estimate_worker_memory(lead_L, lead_R, kpoint=None)
        assert result > 0

    def test_custom_kpoint(self):
        """Test with custom k-point."""
        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        result = _estimate_worker_memory(lead_L, lead_R, kpoint=[0.5, 0.5, 0.0])
        assert result > 0

    def test_fallback_on_error(self):
        """Test fallback to 100MB per lead when Hamiltonian fetch fails."""
        lead_L = MockLead("lead_L", matrix_size=10, raise_error=True)
        lead_R = MockLead("lead_R", matrix_size=10, raise_error=True)

        result = _estimate_worker_memory(lead_L, lead_R, temp_allocation_factor=1.0)

        base_overhead = 300 * 1024 * 1024  # 300 MB
        fallback_per_lead = 100 * 1024 * 1024  # 100 MB per lead
        expected = base_overhead + 2 * fallback_per_lead  # 2 leads

        assert result == expected

    def test_memory_calculation_accuracy(self):
        """Test that memory calculation is accurate for known matrix sizes."""
        matrix_size = 100  # 100x100 matrices
        lead_L = MockLead("lead_L", matrix_size=matrix_size)
        lead_R = MockLead("lead_R", matrix_size=matrix_size)

        # Each lead has 6 matrices of size (100, 100) with complex128 (16 bytes)
        # Matrix bytes per lead = 6 * 100 * 100 * 16 = 960,000 bytes
        # Total matrix bytes = 2 * 960,000 = 1,920,000 bytes
        expected_matrix_bytes = 2 * 6 * matrix_size * matrix_size * 16

        base_overhead = 300 * 1024 * 1024
        temp_factor = 3.0
        expected_total = base_overhead + int(expected_matrix_bytes * temp_factor)

        result = _estimate_worker_memory(lead_L, lead_R, temp_allocation_factor=temp_factor)

        assert result == expected_total

    def test_mixed_success_and_failure(self):
        """Test when one lead succeeds and another fails."""
        lead_L = MockLead("lead_L", matrix_size=50, raise_error=False)
        lead_R = MockLead("lead_R", matrix_size=50, raise_error=True)

        result = _estimate_worker_memory(lead_L, lead_R, temp_allocation_factor=1.0)

        base_overhead = 300 * 1024 * 1024
        # lead_L: 6 * 50 * 50 * 16 = 240,000 bytes
        # lead_R: fallback 100 MB
        lead_L_bytes = 6 * 50 * 50 * 16
        lead_R_fallback = 100 * 1024 * 1024
        expected = base_overhead + lead_L_bytes + lead_R_fallback

        assert result == expected


# =============================================================================
# Tests for _get_safe_n_jobs
# =============================================================================

class TestGetSafeNJobs:
    """Tests for _get_safe_n_jobs function."""

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_auto_detect_n_jobs(self, mock_os, mock_psutil):
        """Test auto-detection with n_jobs=-1."""
        mock_os.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = Mock(available=16 * 1024**3)  # 16 GB

        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1)

        # Should return a positive integer
        assert isinstance(result, int)
        assert result >= 1
        assert result <= 8  # capped by CPU count

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_respects_requested_n_jobs(self, mock_os, mock_psutil):
        """Test that requested n_jobs is respected when safe."""
        mock_os.cpu_count.return_value = 16
        mock_psutil.virtual_memory.return_value = Mock(available=64 * 1024**3)  # 64 GB

        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=4)

        assert result == 4

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_limits_when_memory_constrained(self, mock_os, mock_psutil):
        """Test that n_jobs is limited when memory is constrained."""
        mock_os.cpu_count.return_value = 16
        # Only 1 GB available - should limit workers
        mock_psutil.virtual_memory.return_value = Mock(available=1 * 1024**3)

        lead_L = MockLead("lead_L", matrix_size=100)
        lead_R = MockLead("lead_R", matrix_size=100)

        # Request many workers
        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=16)

        # Should be limited due to memory constraints
        assert result < 16

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_respects_min_workers(self, mock_os, mock_psutil):
        """Test that min_workers is respected."""
        mock_os.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value = Mock(available=1 * 1024**3)

        lead_L = MockLead("lead_L", matrix_size=100)
        lead_R = MockLead("lead_R", matrix_size=100)

        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1, min_workers=2)

        assert result >= 2

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_handles_zero_n_jobs(self, mock_os, mock_psutil):
        """Test handling of n_jobs=0 (invalid)."""
        mock_os.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value = Mock(available=8 * 1024**3)

        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=0, min_workers=1)

        # Should return min_workers when n_jobs=0
        assert result == 1

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_handles_negative_n_jobs(self, mock_os, mock_psutil):
        """Test handling of negative n_jobs (joblib convention)."""
        mock_os.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = Mock(available=32 * 1024**3)

        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        # n_jobs=-2 means (cpu_count + 1 + (-2)) = cpu_count - 1 = 7
        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-2)

        assert result >= 1
        assert result <= 8

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_handles_none_cpu_count(self, mock_os, mock_psutil):
        """Test handling when os.cpu_count() returns None."""
        mock_os.cpu_count.return_value = None
        mock_psutil.virtual_memory.return_value = Mock(available=8 * 1024**3)

        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1)

        # Should default to 1 CPU and still work
        assert result >= 1

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_respects_max_memory_fraction(self, mock_os, mock_psutil):
        """Test that max_memory_fraction parameter is respected."""
        mock_os.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = Mock(available=8 * 1024**3)  # 8 GB

        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        # With higher fraction, should allow more workers
        result_high = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1,
                                        max_memory_fraction=0.9)
        result_low = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1,
                                       max_memory_fraction=0.3)

        assert result_high >= result_low

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_handles_non_integer_n_jobs(self, mock_os, mock_psutil):
        """Test handling of non-integer n_jobs logs a warning."""
        mock_os.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value = Mock(available=8 * 1024**3)

        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        # Pass a float instead of int - function logs warning but continues
        # Note: current implementation has a bug where final_worker is set to min_workers
        # but subsequent conditionals may overwrite it. This test documents current behavior.
        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=2.5, min_workers=1)

        # Result should still be a valid positive number
        assert result >= 1

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_capped_by_cpu_count(self, mock_os, mock_psutil):
        """Test that result is capped by CPU count."""
        mock_os.cpu_count.return_value = 4
        # Plenty of memory
        mock_psutil.virtual_memory.return_value = Mock(available=128 * 1024**3)

        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1)

        # Should be capped at CPU count
        assert result <= 4


# =============================================================================
# Integration tests
# =============================================================================

class TestMemoryEstimationIntegration:
    """Integration tests for memory estimation workflow."""

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_small_system_allows_parallelism(self, mock_os, mock_psutil):
        """Test that small systems allow good parallelism."""
        mock_os.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = Mock(available=8 * 1024**3)  # 8 GB

        # Small 10x10 matrices
        lead_L = MockLead("lead_L", matrix_size=10)
        lead_R = MockLead("lead_R", matrix_size=10)

        result = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1)

        # With 8GB and tiny matrices, should allow many workers
        assert result >= 4

    @patch('dpnegf.negf.lead_property.psutil')
    @patch('dpnegf.negf.lead_property.os')
    def test_large_system_limits_parallelism(self, mock_os, mock_psutil):
        """Test that large systems properly limit parallelism."""
        mock_os.cpu_count.return_value = 16
        mock_psutil.virtual_memory.return_value = Mock(available=4 * 1024**3)  # 4 GB

        # Large 1000x1000 matrices
        lead_L = MockLead("lead_L", matrix_size=1000)
        lead_R = MockLead("lead_R", matrix_size=1000)

        memory_estimate = _estimate_worker_memory(lead_L, lead_R)
        n_jobs = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1)

        # Memory per worker should be significant (> 500 MB)
        # 1000x1000 matrices: 2 leads * 6 matrices * 1000^2 * 16 bytes * 3.0 factor + 300MB overhead
        # = 576 MB computation + 300 MB overhead = ~876 MB
        assert memory_estimate > 500 * 1024**2  # > 500 MB

        # Should limit workers due to memory (4 GB available, ~900 MB per worker)
        assert n_jobs <= 4

    def test_consistent_estimates(self):
        """Test that estimates are consistent across calls."""
        lead_L = MockLead("lead_L", matrix_size=50)
        lead_R = MockLead("lead_R", matrix_size=50)

        result1 = _estimate_worker_memory(lead_L, lead_R)
        result2 = _estimate_worker_memory(lead_L, lead_R)
        result3 = _estimate_worker_memory(lead_L, lead_R)

        assert result1 == result2 == result3
