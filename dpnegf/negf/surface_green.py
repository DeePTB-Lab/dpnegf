import numpy as np
import scipy.linalg as SLA
import logging
import torch
from numba import njit, float64, complex128, int64

log = logging.getLogger(__name__)

def selfEnergy(hL, hLL, sL, sLL, ee, hDL=None, sDL=None, etaLead=1e-8, Bulk=False, 
                    E_ref=0.0, dtype=np.complex128, device='cpu', method='Lopez-Sancho'):
    '''calculates the self-energy and surface Green's function for a given  Hamiltonian and overlap matrix.
    
    Parameters
    ----------
    hL
        Hamiltonian matrix for one principal layer in Lead
    hLL
        Hamiltonian matrix between the most nearby principal layers in Lead
    sL
        Overlap matrix for one principal layer in Lead
    sLL
        Overlap matrix between the most nearby principal layers in Lead
    ee
        the given energy
    hDL
        Hamiltonian matrix between the lead and the device.   
    sDL
        Overlap matrix between the lead and the device.
    etaLead
        A small imaginary number that is used to avoid the singularity of the surface Green's function.
    Bulk, optional
        Ignore it, please.
    chemiPot
        the chemical potential of the lead.
    dtype
        the data type of the tensors used in the calculations. 
    device
        The "device" parameter specifies the device on which the calculations will be performed. It can be
        set to 'cpu' for CPU computation or 'cuda' for GPU computation.
    method
        specify the method for calculating the surface Green's function.The available options 
        are "Lopez-Sancho" and any other value will default to "Lopez-Sancho".
    
    Returns
    -------
        two values: Sig and SGF. The former is self-energy and the latter is surface Green's function.
    
    '''
    # 确保输入是NumPy数组
    hL = convert_to_numpy(hL)
    sL = convert_to_numpy(sL)
    hLL = convert_to_numpy(hLL)
    sLL = convert_to_numpy(sLL)
    ee = convert_to_numpy(ee)
    if hDL is not None:
        hDL = convert_to_numpy(hDL)
    if sDL is not None:
        sDL = convert_to_numpy(sDL)
    E_ref = convert_to_numpy(E_ref)


    
    if not isinstance(ee, np.ndarray):
        eeshifted = np.array(ee, dtype=dtype) + E_ref
    else:
        eeshifted = ee + E_ref
    
    eeshifted = eeshifted.item()
    
    if hDL is None:
        ESH = (eeshifted * sL - hL)
        SGF = surface_green(hL, hLL, sL, sLL, eeshifted + 1j * etaLead , method)
        
        if Bulk:
            Sig = np.linalg.inv(SGF)
        else:
            Sig = ESH - np.linalg.inv(SGF)
    else:
        a, b = hDL.shape
        SGF = surface_green(hL, hLL, sL, sLL, eeshifted + 1j * etaLead , method)
        
        Sig = (eeshifted*sDL-hDL) @ SGF[:b,:b] @ (eeshifted*sDL.conj().T-hDL.conj().T)
    
    Sig = torch.tensor(Sig, dtype=torch.complex128, device=device)
    SGF = torch.tensor(SGF, dtype=torch.complex128, device=device)
    
    return Sig, SGF


_numba_available = False


try:
    from numba import njit, complex128, int64, float64
    from numba.types import Tuple
    NumbaReturnType = Tuple((complex128[:,:], int64, float64, float64))

    @njit(NumbaReturnType(complex128[:,:], complex128[:,:], complex128[:,:], complex128[:,:], complex128))
    def _surface_green_numba_core(H, h01, S, s01, ee):
        # 将 PyTorch 的 h10 = h01.conj().T 逻辑转换为 NumPy
        h10 = np.conj(h01.T)
        s10 = np.conj(s01.T)
        alpha, beta = h10 - ee * s10, h01 - ee * s01
        eps = H.copy()
        epss = H.copy()
        
        converged = False
        iteration = 0
        while not converged:
            iteration += 1
            oldeps, oldepss = eps.copy(), epss.copy()
            oldalpha, oldbeta = alpha.copy(), beta.copy()
            tmpa = np.linalg.solve(ee * S - oldeps, oldalpha)
            tmpb = np.linalg.solve(ee * S - oldeps, oldbeta)
            
            alpha = oldalpha @ tmpa
            beta = oldbeta @ tmpb
            eps = oldeps + oldalpha @ tmpb + oldbeta @ tmpa
            epss = oldepss + oldbeta @ tmpa
            LopezConvTest = np.max(np.abs(alpha) + np.abs(beta))

            if LopezConvTest < 1.0e-40:
                # np.linalg.inv() 等价于 PyTorch 的 .inverse()
                gs = np.linalg.inv(ee * S - epss)
                
                test = ee * S - H - (ee * s01 - h01) @ gs @ (ee * s10 - h10)
                myConvTest = np.max(np.abs((test @ gs) - np.eye(H.shape[0], dtype=h01.dtype)))

                if myConvTest < 3.0e-5:
                    converged = True
                    if myConvTest > 1.0e-8:
                        # 返回结果和警告标志
                        return gs, 1, myConvTest, ee.real
                    else:
                        # 返回结果和成功标志
                        return gs, 0, 0, 0
                else:
                    raise ArithmeticError
        
            if iteration >= 101:
                raise RuntimeError
                
        return gs

    _numba_available = True
    log.info("Numba is available and JIT functions are compiled.")

except (ImportError, Exception) as e:
    log.warning(f"Numba acceleration is not available. Falling back to pure NumPy. Error: {e}")
    _numba_available = False

# NumPy-based implementation of the surface Green's function calculation
def _surface_green_numpy_core(H, h01, S, s01, ee):
    h10 = np.conj(h01.T)
    s10 = np.conj(s01.T)
    alpha, beta = h10 - ee * s10, h01 - ee * s01
    
    eps, epss = H.copy(), H.copy()
    
    converged = False
    iteration = 0
    
    while not converged:
        iteration += 1
        oldeps, oldepss = eps.copy(), epss.copy()
        oldalpha, oldbeta = alpha.copy(), beta.copy()
        tmpa = np.linalg.solve(ee * S - oldeps, oldalpha)
        tmpb = np.linalg.solve(ee * S - oldeps, oldbeta)
        
        alpha = oldalpha @ tmpa
        beta = oldbeta @ tmpb
        eps = oldeps + oldalpha @ tmpb + oldbeta @ tmpa
        epss = oldepss + oldbeta @ tmpa
        
        LopezConvTest = np.max(np.abs(alpha) + np.abs(beta))

        if LopezConvTest < 1.0e-40:
            gs = np.linalg.inv(ee * S - epss)
            
            test = ee * S - H - (ee * s01 - h01) @ gs @ (ee * s10 - h10)
            myConvTest = np.max(np.abs((test @ gs) - np.eye(H.shape[0], dtype=h01.dtype)))
            
            if myConvTest < 3.0e-5:
                converged = True
                if myConvTest > 1.0e-8:
                    log.warning(f"Lopez-scheme not-so-well converged at E = {ee.real:.4f} eV: {myConvTest}")
            else:
                log.error(f"Lopez-Sancho {myConvTest:.8f} Error: gs iteration {iteration}")
                raise ArithmeticError("Criteria not met. Please check output...")
        
        if iteration >= 101:
            log.error("Lopez-scheme not converged after 100 iteration.")
            raise RuntimeError("Lopez-scheme not converged.")
            
    return gs


def surface_green(H, h01, S, s01, ee, 
                  method='Lopez-Sancho',
                  numba_jit=True):
    '''calculate surface green function
    At this stage, we realized Lopez-Sancho scheme and  GEP scheme.
    However, GEP scheme is not so stable, and we strongly recommended  to implement the Lopez-Sancho scheme.

    '''

    if method == 'GEP':
        gs = calcg0(ee, H, S, h01, s01)
        return gs
    else: # Lopez-Sancho scheme
        if numba_jit and _numba_available:
            try:
                # check
                # 1. type check
                assert isinstance(H, np.ndarray), "H must be a NumPy array."
                assert isinstance(h01, np.ndarray), "h01 must be a NumPy array."
                assert isinstance(S, np.ndarray), "S must be a NumPy array."
                assert isinstance(s01, np.ndarray), "s01 must be a NumPy array."
                assert isinstance(ee, (complex, float, int)), "ee must be a complex, float, or integer scalar."

                # 2. dimension check
                assert H.ndim == 2, "H must be a 2D array."
                assert h01.ndim == 2, "h01 must be a 2D array."
                assert S.ndim == 2, "S must be a 2D array."
                assert s01.ndim == 2, "s01 must be a 2D array."

                # 3. complex type check
                assert np.iscomplexobj(H), "H must be a complex array."
                assert np.iscomplexobj(h01), "h01 must be a complex array."
                assert np.iscomplexobj(S), "S must be a complex array."
                assert np.iscomplexobj(s01), "s01 must be a complex array."
                assert isinstance(ee, complex), "ee must be a complex scalar."
                gs, conv_flag, conv_test, e_real = _surface_green_numba_core(H, h01, S, s01, ee)
                if conv_flag == 1:
                    log.warning(f"Lopez-Sancho scheme not-so-well converged at E = {e_real:.4f} eV: {conv_test}")
                return gs
            except (RuntimeError, ArithmeticError) as e:
                log.error(f"Numba JIT function failed at runtime. Falling back to NumPy. Error: {e}")
                return _surface_green_numpy_core(H, h01, S, s01, ee)
        else:
            return _surface_green_numpy_core(H, h01, S, s01, ee)
                



def calcg0(ee, h00, s00, h01, s01):
    '''The `calcg0` function calculates the surface Green's function for a specific |k> , ref. Euro Phys J B 62, 381 (2008)
        Inverse of : NOTE, setup for "right" lead.
        e-h00 -h01  ...
        -h10  e-h11 ...
         .
         .
         .

    Parameters
    ----------
    ee
        The parameter `ee` represents the energy value for which the surface Green's function is
    calculated. It is a complex number that determines the energy of the state being considered.
    h00
        hamiltonian matrix within principal layer
    s00
        overlap matrix within principal layer
    h01
        hamiltonian matrix between two adject principal layers
    s01
        overlap matrix between two adject principal layers
    
    Returns
    -------
        Surface Green's function `g00`.
    
    ''' 
    
    NN = h00.shape[0]
    ee = ee.real + max(ee.imag, 1e-8) * 1.0j

    # Solve generalized eigen-problem
    a, b = np.zeros((2 * NN, 2 * NN), dtype=h00.dtype), np.zeros((2 * NN, 2 * NN), dtype=h00.dtype)
    
    a[0:NN, 0:NN] = ee * s00 - h00
    a[0:NN, NN:2 * NN] = -np.eye(NN, dtype=h00.dtype)
    a[NN:2 * NN, 0:NN] = h01.conj().T - ee * s01.conj().T
    b[0:NN, 0:NN] = h01 - ee * s01
    b[NN:2 * NN, NN:2 * NN] = np.eye(NN, dtype=h00.dtype)


    ev, evec = SLA.eig(a=a, b=b)

    ipiv = np.where(np.abs(ev) < 1.)[0]
    ev, evec = ev[ipiv], evec[:NN, ipiv].T

    # Normalize evec
    norm = np.sqrt(np.diag(evec @ evec.conj().T)) 
    evec = np.diag(1.0 / norm) @ evec

    # E^+ Lambda_+ (E^+)^-1 --->>> g00
    EP = evec.T
    FP = EP @ np.diag(ev) @ np.linalg.inv(EP.conj().T @ EP) @ EP.conj().T
    g00 = np.linalg.inv(ee * s00 - h00 - (h01 - ee * s01) @ FP)
    
    g00 = iterative_gf_numpy(ee, g00, h00, h01, s00, s01, iter=3)

    err = np.max(np.abs(g00 - np.linalg.inv(ee * s00 - h00 - \
                                            (h01 - ee * s01) @ g00 @ (h01.conj().T - ee * s01.conj().T))))
    if err > 1.0e-8:
        print("WARNING: not-so-well converged for RIGHT electrode at E = {0} eV:".format(ee.real), err)
    
    return g00

def iterative_gf_numpy(ee, gs, h00, h01, s00, s01, iter=1):
    '''
    NumPy-based rewrite of the PyTorch iterative_gf function.
    '''
    # 将输入张量转换为 NumPy 数组
    gs = np.array(gs)
    h00 = np.array(h00)
    h01 = np.array(h01)
    s00 = np.array(s00)
    s01 = np.array(s01)
    ee = np.array(ee)
    
    for i in range(iter):
        gs_new = ee*s00 - h00 - (ee * s01 - h01) @ gs @ (ee * s01.conj().T - h01.conj().T)
        gs = np.linalg.pinv(gs_new)
    
    return gs

def iterative_simple_numpy(ee, h00, h01, s00, s01, iter_max=1000):
    '''
    NumPy-based rewrite of the PyTorch iterative_simple function.
    '''
    # 将输入张量转换为 NumPy 数组
    h00 = np.array(h00)
    h01 = np.array(h01)
    s00 = np.array(s00)
    s01 = np.array(s01)
    ee = np.array(ee)
    
    gs = np.linalg.inv(ee*s00 - h00)
    diff_gs = 1
    iteration = 0
    while diff_gs > 1e-8:
        iteration += 1
        gs_prev = gs.copy()
        
        term = (ee * s01 - h01) @ gs_prev @ (ee * s01.conj().T - h01.conj().T)
        gs = np.linalg.inv(ee*s00 - h00 - term)
        
        diff_gs = np.max(np.abs(gs - gs_prev))
        
        if iteration > iter_max:
            log.warning("iterative_simple not converged after 1000 iteration.")
            break
            
    return gs

def convert_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, float):
        return np.array(data)
