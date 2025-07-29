from scipy.sparse.linalg import LinearOperator
from sparse_dot_mkl._mkl_interface._cfunctions import (
    MKL,
    mkl_library_name,
    mkl_get_version_string,
    mkl_get_max_threads
)

from sparse_dot_mkl._mkl_interface._constants import (
    LAYOUT_CODE_C,
    LAYOUT_CODE_F,
    SPARSE_INDEX_BASE_ZERO,
    RETURN_CODES,
    ILP64_MSG,
    SPARSE_MATRIX_TYPE_HERMITIAN,
    SPARSE_MATRIX_TYPE_SYMMETRIC,
    SPARSE_FILL_MODE_LOWER,
    SPARSE_FILL_MODE_UPPER,
    SPARSE_FILL_MODE_FULL,
    SPARSE_DIAG_NON_UNIT,
    SPARSE_DIAG_UNIT
)

from sparse_dot_mkl._mkl_interface._structs import (
    sparse_matrix_t,
    matrix_descr,
    MKL_Complex8,
    MKL_Complex16
)

from sparse_dot_mkl._mkl_interface._common import (
    _check_return_value,
    _out_matrix,
    _get_numpy_layout,
    _export_mkl,
)

import numpy as np
import scipy.sparse as sp
import ctypes as _ctypes



class MKLQuantumOperator(LinearOperator):
    def __init__(self, data, indices, indptr, shape, upper=True, unit=False):
        super(MKLQuantumOperator, self).__init__(shape=shape, dtype=np.complex128)

        ref = sparse_matrix_t()
        
        ret_val = MKL._mkl_sparse_z_create_csr(
            _ctypes.byref(ref),
            _ctypes.c_int(SPARSE_INDEX_BASE_ZERO),
            MKL.MKL_INT(shape[0]),
            MKL.MKL_INT(shape[1]),
            indptr[0:-1],
            indptr[1:],
            indices,
            data,
        )

        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.upper = upper

        self.ref = ref

        # Check return
        _check_return_value(ret_val, MKL._mkl_sparse_z_create_csr.__name__)

        if unit:
            unit_flag = SPARSE_DIAG_UNIT
        else:
            unit_flag = SPARSE_DIAG_NON_UNIT
        
        if upper:
            upper_flag = SPARSE_FILL_MODE_UPPER
        else:
            upper_flag = SPARSE_FILL_MODE_FULL
            

        self.descr = matrix_descr(
            sparse_matrix_type_t=SPARSE_MATRIX_TYPE_HERMITIAN,
            sparse_fill_mode_t=upper_flag,
            sparse_diag_type_t=unit_flag
        )

    def _matvec(self, v, out=None):
        
        if not v.flags.contiguous:
            raise ValueError("vector v is not contiguous")

        output_shape = (self.shape[0],) if v.ndim == 1 else (self.shape[0], 1)

        out = _out_matrix(
            output_shape,
            np.cdouble,
            out_arr=out,
            out_t=False
        )
        ret_val = MKL._mkl_sparse_z_mv(
            10,
            MKL_Complex16(1.),
            self.ref,
            self.descr,
            v,
            MKL_Complex16(0.),
            out,
        )

        _check_return_value(ret_val, MKL._mkl_sparse_z_mv.__name__)
        
        return out

    def _matmat(self, V, out=None):
        output_shape = (self.shape[0], V.shape[1])

        layout_b, ld_b = _get_numpy_layout(V, out)

        output_order = "C" if layout_b == LAYOUT_CODE_C else "F"

        out = _out_matrix(
            output_shape,
            np.cdouble,
            output_order,
            out_arr=out,
            out_t=False
        )

        ret_val = MKL._mkl_sparse_z_mm(
            10,
            MKL_Complex16(1.),
            self.ref,
            self.descr,
            layout_b,
            V,
            output_shape[1],
            ld_b,
            MKL_Complex16(0.),
            out.ctypes.data_as(_ctypes.POINTER(_ctypes.c_double)),
            ld_b,

        )

        _check_return_value(ret_val, MKL._mkl_sparse_z_mm.__name__)

        return out
    
    @classmethod
    def from_csr(cls, csr_mat, upper=True, unit=False):
        if upper:
            mat = sp.triu(csr_mat, format="csr")
        else:
            mat = csr_mat

        return cls(
            mat.data,
            mat.indices,
            mat.indptr,
            mat.shape,
            upper,
            unit
        )
    
    def to_csr(self):
        mat = sp.csr_matrix((self.data, self.indices, self.indptr), shape=self.shape, copy=False)
        if self.upper:
            mat += sp.triu(mat, k=1, format="csr").conj().T
        
        return mat