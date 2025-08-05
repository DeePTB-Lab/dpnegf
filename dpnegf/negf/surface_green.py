import numpy as np
import scipy.linalg as SLA
import logging

log = logging.getLogger(__name__)

def selfEnergy(hL, hLL, sL, sLL, ee, hDL=None, sDL=None, etaLead=1e-8, Bulk=False, 
                    E_ref=0.0, dtype=np.complex128, device='cpu', method='Lopez-Sancho'):
    '''
    Calculates the self-energy and surface Green's function for a given Hamiltonian and overlap matrix.
    NumPy-based rewrite of the PyTorch selfEnergy function.
    '''
    # 确保输入是NumPy数组
    hL = np.array(hL)
    sL = np.array(sL)
    
    # 处理 ee
    if not isinstance(ee, np.ndarray):
        eeshifted = np.array(ee, dtype=dtype) + E_ref
    else:
        eeshifted = ee + E_ref
    
    # 添加一个小的虚部以避免奇点
    eeshifted = eeshifted + 1j * etaLead
    
    if hDL is None:
        ESH = (eeshifted * sL - hL)
        # 调用 NumPy 版本的 surface_green_numpy
        SGF = surface_green(hL, hLL, sL, sLL, eeshifted, method)
        
        if Bulk:
            Sig = np.linalg.inv(SGF)
        else:
            Sig = ESH - np.linalg.inv(SGF)
    else:
        hDL = np.array(hDL)
        sDL = np.array(sDL)
        a, b = hDL.shape
        SGF = surface_green(hL, hLL, sL, sLL, eeshifted, method)
        
        Sig = (eeshifted*sDL-hDL) @ SGF[:b,:b] @ (eeshifted*sDL.conj().T-hDL.conj().T)
    
    return Sig, SGF


def surface_green(H, h01, S, s01, ee, method='Lopez-Sancho'):
    '''
    Calculate surface green function using NumPy.

    This function is a NumPy-based rewrite of the PyTorch SurfaceGreen.forward method.
    '''
    
    # 将输入的张量转换为NumPy数组
    H = np.array(H)
    h01 = np.array(h01)
    S = np.array(S)
    s01 = np.array(s01)
    ee = np.array(ee)

    # 确保 ee 是一个复数，以便处理复数运算
    if not np.iscomplexobj(ee):
        ee = np.complex128(ee)

    if method == 'GEP':
        # 调用 NumPy 版本的 calcg0
        gs = calcg0_numpy(ee, H, S, h01, s01)
    else:
        h10 = h01.conj().T
        s10 = s01.conj().T
        alpha, beta = h10 - ee * s10, h01 - ee * s01
        eps, epss = H.copy(), H.copy()
        
        converged = False
        iteration = 0
        while not converged:
            iteration += 1
            oldeps, oldepss = eps.copy(), epss.copy()
            oldalpha, oldbeta = alpha.copy(), beta.copy()
            
            # 使用 numpy.linalg.solve 替换 torch.linalg.solve
            tmpa = np.linalg.solve(ee * S - oldeps, oldalpha)
            tmpb = np.linalg.solve(ee * S - oldeps, oldbeta)

            # 使用 @ 运算符进行矩阵乘法
            alpha, beta = oldalpha @ tmpa, oldbeta @ tmpb
            eps = oldeps + oldalpha @ tmpb + oldbeta @ tmpa
            epss = oldepss + oldbeta @ tmpa
            
            LopezConvTest = np.max(np.abs(alpha) + np.abs(beta))

            if iteration == 101:
                log.error("Lopez-scheme not converged after 100 iteration.")
                raise RuntimeError("Lopez-scheme not converged.")

            if LopezConvTest < 1.0e-40:
                # 使用 numpy.linalg.inv 替换 tensor.inverse()
                gs = np.linalg.inv(ee * S - epss)

                test = ee * S - H - (ee * s01 - h01) @ gs @ (ee * s10 - h10)
                myConvTest = np.max(np.abs(test @ gs - np.eye(H.shape[0], dtype=H.dtype)))
                
                if myConvTest < 3.0e-5:
                    converged = True
                    if myConvTest > 1.0e-8:
                        log.warning("Lopez-scheme not-so-well converged at E = %.4f eV:" % ee.real.item() + str(myConvTest.item()))
                else:
                    log.error("Lopez-Sancho %.8f " % myConvTest.item() +
                                "Error: gs iteration {0}".format(iteration))
                    raise ArithmeticError("Criteria not met. Please check output...")
                
    return gs


def calcg0_numpy(ee, h00, s00, h01, s01):
    '''
    The `calcg0_numpy` function calculates the surface Green's function for a specific |k> , ref. Euro Phys J B 62, 381 (2008)
    NumPy-based rewrite of the PyTorch calcg0 function.
    ''' 
    # 将输入张量转换为 NumPy 数组
    h00 = np.array(h00)
    s00 = np.array(s00)
    h01 = np.array(h01)
    s01 = np.array(s01)
    ee = np.array(ee)
    
    NN = h00.shape[0]
    ee = ee.real + max(ee.imag, 1e-8) * 1.0j

    # Solve generalized eigen-problem
    a, b = np.zeros((2 * NN, 2 * NN), dtype=h00.dtype), np.zeros((2 * NN, 2 * NN), dtype=h00.dtype)
    
    a[0:NN, 0:NN] = ee * s00 - h00
    a[0:NN, NN:2 * NN] = -np.eye(NN, dtype=h00.dtype)
    a[NN:2 * NN, 0:NN] = h01.conj().T - ee * s01.conj().T
    b[0:NN, 0:NN] = h01 - ee * s01
    b[NN:2 * NN, NN:2 * NN] = np.eye(NN, dtype=h00.dtype)

    # 使用 scipy.linalg.eig 替换 PyTorch 版本
    ev, evec = SLA.eig(a=a, b=b)
    
    # 选择 ev 绝对值小于 1 的特征值及其对应的特征向量
    ipiv = np.where(np.abs(ev) < 1.)[0]
    ev, evec = ev[ipiv], evec[:, ipiv]

    # Normalize evec
    norm = np.sqrt(np.diag(evec.conj().T @ evec))
    evec = evec @ np.diag(1.0 / norm)

    # E^+ Lambda_+ (E^+)^-1 --->>> g00
    EP = evec
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