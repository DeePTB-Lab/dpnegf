import torch
from typing import List
from dpnegf.negf.surface_green import selfEnergy
import logging
import os
from dpnegf.utils.constants import Boltzmann, eV2J
import numpy as np
from dpnegf.negf.bloch import Bloch
import ase
from joblib import Parallel, delayed
import h5py
import glob
import psutil


log = logging.getLogger(__name__)

# """The data output of the intermidiate result should be like this:
# {each kpoint
#     "e_mesh":[],
#     "emap":[]
#     "se":[se(e0), se(e1),...], 
#     "sgf":[...e...]
# }
# There will be a kmap outside like: {(0,0,0):1, (0,1,2):2}, to locate which file it is to reads.
# """


class LeadProperty(object):
    '''
    The Lead class represents a lead in a structure and provides methods for calculating the self energy
    and gamma for the lead.

    Property
    ----------
    hamiltonian
        hamiltonian of the whole structure.
    structure
        structure of the lead.
    tab
        lead tab.
    voltage
        voltage of the lead.
    results_path
        output  path.
    kBT
        Boltzmann constant times temperature.
    efermi
        Fermi energy.
    mu
        chemical potential of the lead.
    gamma
        the broadening function of the isolated energy level of the device
    HL 
        hamiltonian within principal layer
    HLL 
        hamiiltonian between two adjacent principal layers
    HDL 
        hamiltonian between principal layer and device
    SL SLL and SDL 
        the overlap matrix, with the same meaning as HL HLL and HDL.
    

    Method
    ----------
    self_energy
        calculate  the self energy and surface green function at the given kpoint and energy.
    sigma2gamma
        calculate the Gamma function from the self energy.

    '''
    def __init__(self, tab, hamiltonian, structure, results_path, voltage,
                 structure_leads_fold:ase.Atoms=None,bloch_sorted_indice:torch.Tensor=None, useBloch: bool=False,
                    bloch_factor: List[int]=[1,1,1],bloch_R_list:List=None,
                    e_T=300, efermi:float=0.0, E_ref:float=None) -> None:
        self.hamiltonian = hamiltonian
        self.structure = structure
        self.tab = tab
        self.voltage = voltage
        self.results_path = results_path
        self.kBT = Boltzmann * e_T / eV2J
        self.e_T = e_T
        self.efermi = efermi
        if E_ref is None:
            self.E_ref = efermi
        else:
            self.E_ref = E_ref
        self.chemiPot_lead = efermi - voltage
        self.kpoint = None
        self.voltage_old = None
        
        
        self.useBloch = useBloch
        self.bloch_factor = bloch_factor
        self.bloch_sorted_indice = bloch_sorted_indice
        self.bloch_R_list = bloch_R_list
        self.structure_leads_fold = structure_leads_fold
        if self.useBloch:
            assert self.bloch_sorted_indice is not None
            assert self.bloch_R_list is not None
            assert self.bloch_factor is not None
            assert self.structure_leads_fold is not None

    def self_energy(self, kpoint, energy, 
                    eta_lead: float=1e-5,
                    method: str="Lopez-Sancho",
                    save_path: str=None, 
                    save_format: str="h5",
                    se_info_display: bool=False,
                    HS_inmem: bool=True):
        '''calculate and loads the self energy and surface green function at the given kpoint and energy.
        
        Parameters
        ----------
        kpoint
            the coordinates of a specific point in the Brillouin zone. 
        energy
            specific energy value.
        eta_lead : 
            the broadening parameter for calculating lead surface green function.
        method : 
            specify the method for calculating the self energy. At this stage it only supports "Lopez-Sancho".
        save :
            whether to save the self energy. 
        save_path :
            the path to save the self energy. If not specified, the self energy will be saved in the results_path.
        se_info_display :
            whether to display the information of the self energy calculation.   
        HS_inmem :
            whether to store the Hamiltonian and overlap matrix in memory. Default is False.     
        '''
        assert len(np.array(kpoint).reshape(-1)) == 3
        # according to given kpoint and e_mesh, calculating or loading the self energy and surface green function to self.
        if not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy) # Energy relative to Ef
        
        save_path = self._get_save_path(kpoint, energy, save_format, save_path)
        # log.info(f"Self energy save path: {save_path}")

        # Try load
        if os.path.isfile(save_path):
            if se_info_display:
                log.info(f"Loading {self.tab} self-energy from {save_path}")
            self.se = self._load_self_energy(save_path, kpoint, energy, save_format)
            return
        
        # If not loaded, just compute
        if se_info_display:
            log.info(f"Computing {self.tab} self-energy (method={method}) "
                    f"at k={kpoint}, E={energy.item():.6f}")

        self.se = self.self_energy_cal( kpoint, 
                                        energy,
                                        eta_lead=eta_lead, 
                                        method=method,
                                        HS_inmem=HS_inmem)

    def _get_save_path(self, kpoint, energy, save_format: str, save_path: str = None):
        """
        Generate the save path for self-energy files.

        Parameters
        ----------
        kpoint : array-like
            The k-point (length 3).
        energy : torch.Tensor or float
            Energy value.
        save_format : str
            File format, supports "pth" or "h5".
        save_path : str, optional
            User-specified save path. If None, use default under results_path/self_energy.

        Returns
        -------
        str
            Full path to the save file.
        """
        # Ensure kpoint is array for string formatting
        kx, ky, kz = np.asarray(kpoint, dtype=float).reshape(3)
        energy_val = energy.item() if hasattr(energy, "item") else float(energy)

        # Case 1: User provided save_path
        if save_path is not None:
            # If it's a directory, append default filename
            if os.path.isdir(save_path):
                if save_format == "pth":
                    return os.path.join(save_path,
                                        f"se_{self.tab}_k{kx:.4f}_{ky:.4f}_{kz:.4f}_E{energy_val:.6f}.pth")
                elif save_format == "h5":
                    if self.tab == "lead_L":
                        return os.path.join(save_path, "self_energy_leadL.h5")
                    elif self.tab == "lead_R":
                        return os.path.join(save_path, "self_energy_leadR.h5")
                else:
                    raise ValueError(f"Unsupported save_format {save_format}")
            return save_path  # direct file path given by user

        # Case 2: Default path under results_path
        parent_dir = os.path.join(self.results_path, "self_energy")
        os.makedirs(parent_dir, exist_ok=True)

        if save_format == "pth":
            return os.path.join(parent_dir,
                                f"se_{self.tab}_k{kx:.4f}_{ky:.4f}_{kz:.4f}_E{energy_val:.6f}.pth")

        elif save_format == "h5":
            if self.tab == "lead_L":
                return os.path.join(parent_dir, "self_energy_leadL.h5")
            elif self.tab == "lead_R":
                return os.path.join(parent_dir, "self_energy_leadR.h5")
            else:
                raise ValueError(f"Unsupported tab {self.tab} for h5 save.")

        else:
            raise ValueError(f"Unsupported save_format {save_format}, only 'pth' and 'h5' are supported.")

    @staticmethod
    def _load_self_energy(save_path: str, kpoint, energy, save_format: str):
        """
        Load self-energy from file.

        Parameters
        ----------
        save_path : str
            Path to the saved self-energy file.
        kpoint : array-like
            The k-point (length 3).
        energy : torch.Tensor or float
            Energy value.
        save_format : str
            File format, supports "pth" or "h5".

        Returns
        -------
        torch.Tensor
            Loaded self-energy tensor.
        """
        if save_format == "pth":
            try:
                se = torch.load(save_path, weights_only=False)
            except Exception as e:
                raise IOError(f"Failed to load self-energy from {save_path} (pth format).") from e

        elif save_format == "h5":
            try:
                data = read_from_hdf5(save_path, kpoint, energy)
                se = torch.as_tensor(data, dtype=torch.complex128)  # 自能一般是复数
            except KeyError as e:
                kx, ky, kz = np.asarray(kpoint, dtype=float).reshape(3)
                ev = energy.item() if hasattr(energy, "item") else float(energy)
                raise KeyError(
                    f"Cannot find self-energy in {save_path} "
                    f"for k=({kx:.4f},{ky:.4f},{kz:.4f}), E={ev:.6f}"
                ) from e
            except Exception as e:
                raise IOError(f"Failed to read HDF5 self-energy from {save_path}.") from e

        else:
            raise ValueError(f"Unsupported save_format {save_format}, only 'pth' and 'h5' are supported.")

        return se


    def self_energy_cal(self, 
                        kpoint, 
                        energy, 
                        eta_lead: float=1e-5,
                        method: str="Lopez-Sancho",
                        HS_inmem: bool=True):
        """
        Calculates the self-energy for a lead in a quantum transport calculation.
        This method computes the self-energy matrix for a given k-point and energy, 
        using either the standard or Bloch-based approach depending on the object's configuration.
        Parameters
        ----------
        kpoint : array-like
            The k-point in reciprocal space at which to calculate the self-energy.
        energy : float or torch.Tensor
            The energy value at which to evaluate the self-energy.
        eta_lead : float, optional
            Small imaginary part added to the energy for numerical stability (default: 1e-5).
        method : str, optional
            The method used for self-energy calculation (default: "Lopez-Sancho").
        HS_inmem : bool, optional
            If False, deletes Hamiltonian and overlap matrices from memory after calculation (default: True).
            This is useful for large systems to save memory.
        Returns
        -------
        se : torch.Tensor
            The calculated self-energy matrix for the specified k-point and energy.
        Notes
        -----
        - If `useBloch` is True, the calculation unfolds the k-points and applies Bloch phase factors.
        - The method caches Hamiltonian and overlap matrices for efficiency unless `HS_inmem` is False.
        - The shape of the returned self-energy matrix is consistent with the reduced Hamiltonian blocks.
        """      
        subblocks = self.hamiltonian.get_hs_device(kpoint, only_subblocks=True)
        # calculate self energy
        if not self.useBloch:
            if  not hasattr(self, "HL") \
                or abs(self.voltage_old-self.voltage)>1e-6 \
                or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:

                self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk \
                    = self.hamiltonian.get_hs_lead(kpoint, tab=self.tab, v=self.voltage)
                self.voltage_old = self.voltage
                self.kpoint = torch.tensor(kpoint)

            HDL_reduced, SDL_reduced = self.HDL_reduced(self.HDLk, self.SDLk,subblocks)
            
            self.se, _ = selfEnergy(
                ee=energy,
                hL=self.HLk,
                hLL=self.HLLk,
                sL=self.SLk,
                sLL=self.SLLk,
                hDL=HDL_reduced,
                sDL=SDL_reduced,             #TODO: check chemiPot settiing is correct or not
                E_ref=self.E_ref,
                etaLead=eta_lead, 
                method=method
            )

            # torch.save(self.se, os.path.join(self.results_path, f"se_nobloch_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_{energy}.pth"))
        
        else:
            if not hasattr(self, "HL") \
                or abs(self.voltage_old-self.voltage)>1e-6 \
                or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:
                self.kpoint = torch.tensor(kpoint)
                self.voltage_old = self.voltage

            bloch_unfolder = Bloch(self.bloch_factor)
            kpoints_bloch = bloch_unfolder.unfold_points(self.kpoint.tolist())
            sgf_k = []
            m_size = self.bloch_factor[1]*self.bloch_factor[0]
            for k_bloch in kpoints_bloch:
                k_bloch = torch.tensor(k_bloch)
                self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk \
                    = self.hamiltonian.get_hs_lead(k_bloch, tab=self.tab, v=self.voltage)
                
                _, sgf = selfEnergy(
                    ee=energy,
                    hL=self.HLk,
                    hLL=self.HLLk,
                    sL=self.SLk,
                    sLL=self.SLLk,            #TODO: check chemiPot settiing is correct or not
                    E_ref=self.E_ref,  # temmporarily change to self.efermi for the case in which applying lead bias to corresponding to Nanotcad
                    etaLead=eta_lead, 
                    method=method
                )
                phase_factor_m = torch.zeros([m_size,m_size],dtype=torch.complex128)
                for i in range(m_size):
                    for j in range(m_size):
                        if i == j:
                            phase_factor_m[i,j] = 1
                        else:
                            phase_factor_m[i,j] = torch.exp(torch.tensor(1j)*2*torch.pi*torch.dot(self.bloch_R_list[j]-self.bloch_R_list[i],k_bloch))  
                phase_factor_m = phase_factor_m.contiguous()
                sgf = sgf.contiguous()
                sgf_k.append(torch.kron(phase_factor_m,sgf)) 
             

            sgf_k = torch.sum(torch.stack(sgf_k),dim=0)/len(sgf_k)
            sgf_k = sgf_k[self.bloch_sorted_indice,:][:,self.bloch_sorted_indice]
            b = self.HDLk.shape[1] # size of lead hamiltonian

            # reduce the Hamiltonian and overlap matrix based on the non-zero range of HDL
            HDL_reduced, SDL_reduced = self.HDL_reduced(self.HDLk, self.SDLk,subblocks) 
            if not isinstance(energy, torch.Tensor):
                eeshifted = torch.scalar_tensor(energy, dtype=torch.complex128) + self.E_ref
            else:
                eeshifted = energy + self.E_ref
            self.se = (eeshifted*SDL_reduced-HDL_reduced) @ sgf_k[:b,:b] @ (eeshifted*SDL_reduced.conj().T-HDL_reduced.conj().T)
            # In subblocks case, the self energy shape of left/right lead should be consistent with subblocks[0] and subblocks[-1]
        if not HS_inmem:
            del self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk

        return self.se

    @staticmethod
    def HDL_reduced(HDL: torch.Tensor, SDL: torch.Tensor, subblocks: np.ndarray) -> torch.Tensor:
        '''This function takes in Hamiltonian/Overlap matrix between lead and device and reduces 
        it based on the subblocks results or non-zero range of the Hamiltonian matrix.

            When the device part has only one orbital, the Hamiltonian matrix is not reduced.
        
        Parameters
        ----------
        HDL : torch.Tensor
            HDL is a torch.Tensor representing the Hamiltonian matrix between the first principal layer and the device.
        SDL : torch.Tensor
            SDL is a torch.Tensor representing the overlap matrix between the first principal layer and the device.
        
        Returns
        -------
        HDL_reduced, SDL_reduced
            The reduced Hamiltonian and overlap matrix.
        
        '''
        assert len(HDL.shape) == 2, "The shape of HDL should be 2."
        assert len(SDL.shape) == 2, "The shape of SDL should be 2."
        assert HDL.shape == SDL.shape, "The shape of HDL and SDL should be the same."

        HDL_nonzero_range = (HDL.nonzero().min(dim=0).values, HDL.nonzero().max(dim=0).values)
        if subblocks is None:
            cut_range = HDL_nonzero_range
        else:
            cut_range = ((subblocks[-1],subblocks[-1]), (subblocks[0],subblocks[0]))
        # HDL_nonzero_range is a tuple((min_row,min_col),(max_row,max_col))
        if HDL.shape[0] == 1: # Only 1 orbital in the device
            HDL_reduced = HDL
            SDL_reduced = SDL
        elif HDL_nonzero_range[0][0] > 0: # Right lead
            if subblocks is None:
                HDL_reduced = HDL[cut_range[0][0]:, :]
                SDL_reduced = SDL[cut_range[0][0]:, :]
            else:
                HDL_reduced = HDL[-1*cut_range[0][0]:, :]
                SDL_reduced = SDL[-1*cut_range[0][0]:, :]
        else: # Left lead
            if subblocks is None:
                HDL_reduced = HDL[:cut_range[1][0]+1, :]
                SDL_reduced = SDL[:cut_range[1][0]+1, :]
            else:
                HDL_reduced = HDL[:cut_range[1][0], :]
                SDL_reduced = SDL[:cut_range[1][0], :]

        return HDL_reduced, SDL_reduced


    def sigmaLR2Gamma(self, se):
        '''calculate the Gamma function from the self energy.
        
        Gamma function is the broadening function of the isolated energy level of the device.

        Parameters
        ----------
        se
            The parameter "se" represents self energy, a complex matrix.
        
        Returns
        -------
        Gamma
            The Gamma function, Gamma = 1j(se-se^dagger).
        
        '''
        return 1j * (se - se.conj().T)
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp((x - self.chemiPot_lead)/ self.kBT))
    
    @property
    def gamma(self):
        return self.sigmaLR2Gamma(self.se)


def _estimate_worker_memory(lead_L, lead_R, kpoint=None, temp_allocation_factor=3.0):
    """
    Estimate memory (in bytes) needed per joblib worker for self-energy calculation.

    The estimation separates two components:
    1. Base overhead: Fixed memory for Python process and imported libraries
       (Python interpreter, NumPy, SciPy, PyTorch, DeePTB, etc.)
    2. Computation memory: Dynamic memory for matrices and intermediate calculations,
       scaled by a factor to account for temporary allocations during surface Green's
       function iteration, matrix products, and LAPACK/BLAS workspace.

    Parameters
    ----------
    lead_L, lead_R : LeadProperty
        Lead objects containing Hamiltonian data.
    kpoint : array-like, optional
        A sample k-point to use for fetching Hamiltonian matrices. If None, uses [0, 0, 0].
    temp_allocation_factor : float
        Multiplier for computation memory to account for intermediate arrays created
        during surface Green's function iteration and matrix operations. Default 3.0.

    Returns
    -------
    int
        Estimated memory in bytes per worker.
    """
    # Base overhead for Python process + libraries (interpreter, numpy, scipy, torch, etc.)
    base_overhead = 300 * 1024 * 1024  # 300 MB

    matrix_bytes = 0

    if kpoint is None:
        kpoint = [0.0, 0.0, 0.0]  # use Gamma point if not provided

    # Estimate from lead Hamiltonian matrices using get_hs_lead method
    # Each complex128 element = 16 bytes
    for lead in [lead_L, lead_R]:
        try:
            # get_hs_lead returns: (hL, hLL, hDL, sL, sLL, sDL)
            hL, hLL, hDL, sL, sLL, sDL = lead.hamiltonian.get_hs_lead(
                kpoint=kpoint, tab=lead.tab, v=lead.voltage
            )
            # Sum up memory for all matrices
            for tensor in [hL, hLL, hDL, sL, sLL, sDL]:
                if tensor is not None:
                    if hasattr(tensor, 'numel'):  # PyTorch tensor
                        matrix_bytes += tensor.numel() * 16  # complex128
                    elif hasattr(tensor, 'nbytes'):  # NumPy array
                        matrix_bytes += tensor.nbytes
        except Exception as e:
            log.warning(f"Could not estimate matrix memory from {lead.tab}: {e}"
                        " Using fallback matrix estimate: 100 MB this lead.")
            # Fallback: assume 100 MB per lead
            matrix_bytes += 100 * 1024 * 1024
            
    # Total estimate: base overhead + scaled computation memory
    computation_memory = matrix_bytes * temp_allocation_factor
    total_estimate = base_overhead + int(computation_memory)

    return total_estimate


def _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=-1, max_memory_fraction=0.7, min_workers=1, kpoint=None):
    """
    Calculate safe number of parallel workers based on available system memory.

    Parameters
    ----------
    lead_L, lead_R : LeadProperty
        Lead objects for memory estimation.
    requested_n_jobs : int
        User-requested n_jobs. -1 means auto-detect.
    max_memory_fraction : float
        Maximum fraction of available memory to use. Default 0.7.
    min_workers : int
        Minimum number of workers to use. Default 1.
    kpoint : array-like, optional
        A sample k-point for fetching Hamiltonian matrices to estimate memory.

    Returns
    -------
    int
        Safe number of parallel workers.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        cpu_count = 1
        log.warning("os.cpu_count() returned None or invalid value. Defaulting to 1 CPU core.")

    available_memory = psutil.virtual_memory().available
    memory_per_worker = _estimate_worker_memory(lead_L, lead_R, kpoint=kpoint)

    # Calculate max workers that fit in available memory
    if memory_per_worker <= 0:
        log.warning(f"Memory estimation returned non-positive value. Using min_workers={min_workers}.")
        return min_workers

    # Calculate max workers that fit in available memory
    max_workers_by_memory = int((available_memory * max_memory_fraction) / memory_per_worker)
    max_workers_by_memory = int((available_memory * max_memory_fraction) / memory_per_worker)
    max_workers_by_memory = max(max_workers_by_memory, min_workers)

    # Cap by CPU count
    max_workers = min(max_workers_by_memory, cpu_count)

    safe_n_worker = 0
    # check requested_n_jobs is a number
    if not isinstance(requested_n_jobs, int):
        log.warning(f"Requested n_jobs={requested_n_jobs} is not an integer. \n"
                    f"Using min_workers={min_workers}.")
        safe_n_worker = min_workers

    if requested_n_jobs == -1:
        safe_n_worker = max_workers
    elif requested_n_jobs == 0:
        log.warning(f"Requested n_jobs=0 is invalid. Using min_workers={min_workers}.")
        safe_n_worker = min_workers
    elif requested_n_jobs > 0:
        if requested_n_jobs > max_workers:
            log.warning(f"Requested n_jobs={requested_n_jobs} may exceed available memory. "
                       f"Limiting to {max_workers} workers "
                       f"(available: {available_memory / 1e9:.1f} GB, "
                       f"est. per worker: {memory_per_worker / 1e9:.1f} GB)")
            safe_n_worker = max_workers
        else:
            safe_n_worker = requested_n_jobs
    else:
        # Negative values other than -1: joblib interprets as (cpu_count + 1 + n_jobs)
        effective_n_jobs = max(cpu_count + 1 + requested_n_jobs, min_workers)
        safe_n_worker = min(effective_n_jobs, max_workers)

    log.info(f"Estimated safe n_jobs={safe_n_worker} based on available memory.")
    return safe_n_worker
        


def compute_all_self_energy(eta, lead_L, lead_R, kpoints_grid, energy_grid,
                            self_energy_save_path=None, n_jobs=-1, batch_size=200):
    """
    Computes and saves self-energy matrices for all combinations of k-points and energy values
    for left and right leads.

    The self-energy calculations are performed in parallel batches, and results are saved as HDF5 files.
    Temporary files are merged into final output files for each lead.

    Parameters
    ----------
    eta : float
        Small imaginary part added to energy for numerical stability.
    lead_L : Lead
        lead object containing Left lead Hamiltonian and results path.
    lead_R : Lead
        lead object containing Right lead Hamiltonian and results path.
    kpoints_grid : array-like
        List or array of k-points to compute self-energy for.
    energy_grid : array-like
        List or array of energy values to compute self-energy for.
    self_energy_save_path : str or None, optional
        Directory to save self-energy files. If None, uses lead_L's results_path.
    n_jobs : int, optional
        Number of parallel jobs to use. Default is -1 (use all available CPUs).
    batch_size : int, optional
        Number of (k, e) tasks per parallel batch. Default is 200.

    Returns
    -------
    None
        Results are saved to disk as HDF5 files.
    """
    if self_energy_save_path is None:
        if lead_L.results_path != lead_R.results_path:
            log.warning("The results_path of lead_L and lead_R are different. "
                        "Self energy files will be saved in lead_L's results_path.")
        self_energy_save_path = os.path.join(lead_L.results_path, "self_energy")

    # Calculate safe number of workers based on available memory
    # Use first k-point for memory estimation
    sample_kpoint = kpoints_grid[0] if len(kpoints_grid) > 0 else None
    safe_n_jobs = _get_safe_n_jobs(lead_L, lead_R, requested_n_jobs=n_jobs, kpoint=sample_kpoint)
    if n_jobs == -1:
        log.info(f"Auto-detected safe n_jobs={safe_n_jobs} based on available memory")
    elif safe_n_jobs < n_jobs:
        log.info(f"Adjusted n_jobs from {n_jobs} to {safe_n_jobs} due to memory constraints")

    total_tasks = [(k, e) for k in kpoints_grid for e in energy_grid]
    if len(total_tasks) <= batch_size:
        Parallel(n_jobs=safe_n_jobs, backend="loky")(
            delayed(self_energy_worker)(k, e, eta, lead_L, lead_R, self_energy_save_path)
            for k, e in total_tasks
        )
    else:
        for i in range(0, len(total_tasks), batch_size):
            batch = total_tasks[i:i+batch_size]
            Parallel(n_jobs=safe_n_jobs, backend="loky")(
                delayed(self_energy_worker)(k, e, eta, lead_L, lead_R, self_energy_save_path)
                for k, e in batch
            )


    save_path_L = os.path.join(self_energy_save_path, "self_energy_leadL.h5")
    save_path_R = os.path.join(self_energy_save_path, "self_energy_leadR.h5")

    merge_hdf5_files(self_energy_save_path, save_path_L, pattern="tmp_leadL_*.h5")
    merge_hdf5_files(self_energy_save_path, save_path_R, pattern="tmp_leadR_*.h5")





def self_energy_worker(k, e, eta, lead_L, lead_R, self_energy_save_path):
    """
    Calculates the self-energy for left and right leads at a given k-point and energy,
    and saves the results to HDF5 files.

    Parameters
    ----------
    k : array-like
        The k-point in reciprocal space, typically a 3-element array or list.
    e : float
        The energy value at which to calculate the self-energy.
    eta : float
        A small imaginary part added to the energy for numerical stability.
    lead_L : object
        The left lead object, which must implement a `self_energy_cal` method.
    lead_R : object
        The right lead object, which must implement a `self_energy_cal` method.
    self_energy_save_path : str
        Directory path where the self-energy HDF5 files will be saved.

    Returns
    -------
    None
        The function saves the calculated self-energies to files and does not return anything.
    """

    save_tmp_L = os.path.join(self_energy_save_path, f"tmp_leadL_k{k[0]}_{k[1]}_{k[2]}_E{e:.8f}.h5")
    save_tmp_R = os.path.join(self_energy_save_path, f"tmp_leadR_k{k[0]}_{k[1]}_{k[2]}_E{e:.8f}.h5")

    seL = lead_L.self_energy_cal(kpoint=k, energy=e, eta_lead=eta)
    seR = lead_R.self_energy_cal(kpoint=k, energy=e, eta_lead=eta)

    write_to_hdf5(save_tmp_L, k, e, seL)
    write_to_hdf5(save_tmp_R, k, e, seR)


def write_to_hdf5(h5_path, k, e, se):
    with h5py.File(h5_path, "a") as f:
        group_name = f"E_{e:.8f}"
        dset_name = f"k_{k[0]}_{k[1]}_{k[2]}"
        grp = f.require_group(group_name)
        if dset_name in grp:
            log.warning(f"Dataset {dset_name} already exists in group {group_name}. Skipping it.")
        grp.create_dataset(dset_name, data=se.cpu().numpy(), compression="gzip")
        f.flush()



def read_from_hdf5(h5_path, k, e):
    with h5py.File(h5_path, "r") as f:
        group_name = f"E_{e:.8f}"
        dset_name = f"k_{k[0]}_{k[1]}_{k[2]}"
        if group_name in f and dset_name in f[group_name]:
            return f[group_name][dset_name][:]
        else:
            raise KeyError(f"Data for kpoint {k} and energy {e} not found.")



def merge_hdf5_files(tmp_dir, output_path, pattern, remove=True):

    tmp_paths = sorted(glob.glob(os.path.join(tmp_dir, pattern)))
    if not tmp_paths:
        raise ValueError(f"No files matched pattern '{pattern}' in '{tmp_dir}'")

    log.info(f"Merging {len(tmp_paths)} tmp self energy files into {output_path}")

    with h5py.File(output_path, 'a') as fout:
        for path in tmp_paths:
            with h5py.File(path, 'r') as fin:
                for group_name in fin:
                    fin_group = fin[group_name]
                    fout_group = fout.require_group(group_name)

                    for dset_name in fin_group:
                        if dset_name in fout_group:
                            log.warning(f"Dataset '{dset_name}' already exists in group '{group_name}'. Skipping.")
                            continue
                        fin_group.copy(dset_name, fout_group)

    log.info("Merge complete.")

    if remove:
        for path in tmp_paths:
            try:
                os.remove(path)
                # log.info(f"Deleted tmp file: {path}")
            except Exception as e:
                log.warning(f"Failed to delete {path}: {e}")


def _has_saved_self_energy(root: str) -> bool:
        from pathlib import Path
        p = Path(root) if root is not None else None
        if p is None or not p.exists():
            return False
        
        patterns = ("*.h5", "*.pth")
        for pat in patterns:
            if any(p.rglob(pat)):
                return True
        return False