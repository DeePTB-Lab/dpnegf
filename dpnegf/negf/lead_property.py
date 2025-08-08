import torch
from typing import List
from dpnegf.negf.surface_green import selfEnergy
import logging
from dpnegf.negf.negf_utils import update_kmap, update_temp_file
import os
from dpnegf.utils.constants import Boltzmann, eV2J
import numpy as np
from dpnegf.negf.bloch import Bloch
import torch.profiler
import ase
from joblib import Parallel, delayed
from multiprocessing import Lock
import h5py
import glob

write_lock = Lock()
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
                    save: bool=False, 
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
        
        if save_path is None:
            parent_dir = os.path.join(self.results_path, "self_energy")
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            if save_format == "pth":
                save_path = os.path.join(parent_dir, 
                                         f"se_{self.tab}_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_E{energy}.pth")
            elif save_format == "h5":
                if self.tab == "lead_L":
                    save_path = os.path.join(parent_dir, "self_energy_leadL.h5")
                elif self.tab == "lead_R":
                    save_path = os.path.join(parent_dir, "self_energy_leadR.h5")
                else:
                    raise ValueError(f"Unsupported tab {self.tab} for saving self energy.")
            else:
                raise ValueError(f"Unsupported save format {save_format}. Only 'pth' and 'h5' are supported.")

        # If the file in save_path exists, then directly load it    
        if os.path.exists(save_path):
            if se_info_display: 
                log.info(f"Loading self energy from {save_path}")   

            if os.path.isdir(save_path):
                if save_format == "pth":
                    save_path = os.path.join(save_path, f"se_{self.tab}_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_E{energy}.pth")
                elif save_format == "h5":
                    save_path = os.path.join(save_path, f"self_energy_{self.tab}.h5")
                else:
                    raise ValueError(f"Unsupported save format {save_format}. Only 'pth' and 'h5' are supported.")
                

            assert os.path.exists(save_path), f"Cannot find the self energy file {save_path}"
            if save_path.endswith(".pth"):
                # if the save_path is a directory, then the self energy file is stored in the directory
                self.se = torch.load(save_path, weights_only=False)
            elif save_path.endswith(".h5"):
                try:
                    self.se = read_from_hdf5(save_path, kpoint, energy)
                    self.se = torch.from_numpy(self.se)
                except KeyError as e:
                    log.error(f"Cannot find the self energy for kpoint {kpoint} and energy {energy} in {save_path}.")
                    raise e

            return
            
        else:
            if se_info_display:
                log.info("-"*50)
                log.info(f"Not find stored {self.tab} self energy. Calculating it at kpoint {kpoint} and energy {energy}.")
                log.info("-"*50)
        
        self.self_energy_cal(kpoint, energy, eta_lead=eta_lead, method=method,HS_inmem=HS_inmem)

    def self_energy_cal(self, 
                        kpoint, 
                        energy, 
                        eta_lead: float=1e-5,
                        method: str="Lopez-Sancho",
                        HS_inmem: bool=True):
        
        subblocks = self.hamiltonian.get_hs_device(kpoint, only_subblocks=True)
        # calculate self energy
        if not self.useBloch:
            if not hasattr(self, "HL") or abs(self.voltage_old-self.voltage)>1e-6 or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:
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
    


# def compute_all_self_energy(eta, lead_L, lead_R, kpoints_grid, energy_grid, n_jobs=-1):
#     """
#     Compute the self-energy for all combinations of k-points and energy values in parallel using joblib.
#     Parameters:
#         eta (float): The broadening parameter for calculating lead surface green function.
#         lead_L (LeadProperty): The left lead object.
#         lead_R (LeadProperty): The right lead object.
#         kpoints_grid (Iterable): An iterable of k-point values to compute self-energy for.
#         energy_grid (Iterable): An iterable of energy values to compute self-energy for.
#         n_jobs (int, optional): The number of parallel jobs to run. Defaults to -1 (use all available cores).
#     Notes:
#         This method uses joblib's Parallel to distribute the computation of self-energy across multiple processes.
#         The worker function `self_energy_worker` must be serializable and defined at the top level.
#     """
#     # joblib's Parallel and delayed are used to parallelize the self-energy computation
#     # joblib requires worker function to be top-level or serializable
#     Parallel(n_jobs=n_jobs, backend="loky")(
#         delayed(self_energy_worker)(k, e, eta, lead_L, lead_R)
#         for k in kpoints_grid
#         for e in energy_grid
#     )


def compute_all_self_energy(eta, lead_L, lead_R, kpoints_grid, energy_grid, n_jobs=-1, batch_size=200):

    total_tasks = [(k, e) for k in kpoints_grid for e in energy_grid]
    if len(total_tasks) <= batch_size:
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self_energy_worker)(k, e, eta, lead_L, lead_R)
            for k, e in total_tasks
        )
    
    else:
        for i in range(0, len(total_tasks), batch_size):
            batch = total_tasks[i:i+batch_size]
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self_energy_worker)(k, e, eta, lead_L, lead_R)
                for k, e in batch
            )

    save_path_L = os.path.join(lead_L.results_path, "self_energy", "self_energy_leadL.h5")
    save_path_R = os.path.join(lead_R.results_path, "self_energy", "self_energy_leadR.h5")

    merge_hdf5_files(os.path.join(lead_L.results_path, "self_energy"), save_path_L, pattern="tmp_leadL_*.h5")
    merge_hdf5_files(os.path.join(lead_R.results_path, "self_energy"), save_path_R, pattern="tmp_leadR_*.h5")





def self_energy_worker(k, e, eta, lead_L, lead_R):

    save_tmp_L = os.path.join(lead_L.results_path, "self_energy", f"tmp_leadL_k{k[0]}_{k[1]}_{k[2]}_E{e:.8f}.h5")
    save_tmp_R = os.path.join(lead_R.results_path, "self_energy", f"tmp_leadR_k{k[0]}_{k[1]}_{k[2]}_E{e:.8f}.h5")

    seL = lead_L.self_energy_cal(kpoint=k, energy=e, eta_lead=eta)
    seR = lead_R.self_energy_cal(kpoint=k, energy=e, eta_lead=eta)

    write_to_hdf5(save_tmp_L, k, e, seL)
    write_to_hdf5(save_tmp_R, k, e, seR)


def write_to_hdf5(h5_path, k, e, se):
    with h5py.File(h5_path, "a") as f:
        group_name = f"k_{k[0]}_{k[1]}_{k[2]}"
        dset_name = f"E_{e:.8f}"
        grp = f.require_group(group_name)
        if dset_name in grp:
            log.warning(f"Dataset {dset_name} already exists in group {group_name}. Passing it.")
        grp.create_dataset(dset_name, data=se.cpu().numpy(), compression="gzip")
        f.flush()



def read_from_hdf5(h5_path, kpoint, energy):
    with h5py.File(h5_path, "r") as f:
        group_name = f"k_{kpoint[0]}_{kpoint[1]}_{kpoint[2]}"
        dset_name = f"E_{energy:.8f}"
        if group_name in f and dset_name in f[group_name]:
            return f[group_name][dset_name][:]
        else:
            raise KeyError(f"Data for kpoint {kpoint} and energy {energy} not found.")



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