import torch
from dpnegf.negf.negf_utils import quad, gauss_xw,leggauss,update_kmap
from dpnegf.utils.constants import valence_electron
from dpnegf.negf.ozaki_res_cal import ozaki_residues
from dpnegf.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from dpnegf.utils.elec_struc_cal import ElecStruCal
from dpnegf.negf.density import Ozaki,Fiori
from dpnegf.negf.device_property import DeviceProperty
from dpnegf.negf.lead_property import LeadProperty, compute_all_self_energy, _has_saved_self_energy
from dpnegf.negf.negf_utils import is_fully_covered
import ase
from dpnegf.utils.constants import Boltzmann, eV2J
import numpy as np
from dpnegf.utils.make_kpoints import kmesh_sampling_negf
import logging
import json
from dpnegf.negf.poisson_init import Grid,Interface3D,Dirichlet,Dielectric
from dpnegf.negf.scf_method import PDIISMixer,DIISMixer,BroydenFirstMixer,BroydenSecondMixer,AndersonMixer
from typing import Optional, Union
from dpnegf.utils.tools import apply_gaussian_filter_3d
from pyinstrument import Profiler
import os

log = logging.getLogger(__name__)


try:
    from dptb.data import AtomicData, AtomicDataDict
except ImportError:
    raise ImportError("dptb.data is not available. Please install dptb package to use AtomicData.")
    



# TODO : add common class to set all the dtype and precision.

class NEGF(object):
    def __init__(self, 
                model: torch.nn.Module,
                structure: Union[AtomicData, ase.Atoms, str],
                ele_T: float,
                emin: float, emax: float, espacing: float,
                density_options: dict,
                unit: str,
                scf: bool, poisson_options: dict,
                stru_options: dict,eta_lead: float,eta_device: float,
                block_tridiagonal: bool,
                sgf_solver: str,
                e_fermi: float=None,
                use_saved_HS: bool=False, saved_HS_path: str=None,
                use_saved_se: bool=False, self_energy_save_path: str=None, se_info_display: bool=False,
                out_tc: bool=False,out_dos: bool=False,out_density: bool=False,out_potential: bool=False,
                out_current: bool=False,out_current_nscf: bool=False,out_ldos: bool=False,out_lcurrent: bool=False,
                results_path: Optional[str]=None,
                torch_device: Union[str, torch.device]=torch.device('cpu'),
                AtomicData_options: Optional[dict]=None, 
                **kwargs):
        
        
        # self.model = model # No need to set model as property for memory saving      
        self.results_path = results_path
        self.cdtype = torch.complex128
        self.torch_device = torch_device
               
        # get the parameters
        self.ele_T = ele_T
        self.kBT = Boltzmann * self.ele_T / eV2J # change to eV
        self.e_fermi = e_fermi
        self.eta_lead = eta_lead; self.eta_device = eta_device
        self.emin = emin; self.emax = emax; self.espacing = espacing
        self.stru_options = stru_options
        self.poisson_options = poisson_options
        if e_fermi is None:
            for lead in ["lead_L", "lead_R"]:
                assert "kmesh_lead_Ef" in self.stru_options[lead], f"{lead} must have 'kmesh_lead_Ef' set in stru_options if e_fermi is None"


        self.use_saved_HS = use_saved_HS
        self.saved_HS_path = saved_HS_path

        self.sgf_solver = sgf_solver
        self.use_saved_se = use_saved_se # whether to use the saved self-energy or not
        self.self_energy_save_path = self_energy_save_path # The directory to save the self-energy or for saved self-energy
        self.se_info_display = se_info_display # whether to display the self-energy information after calculation
        self.pbc = self.stru_options["pbc"]

        if  self.stru_options["lead_L"]["useBloch"] or self.stru_options["lead_R"]["useBloch"]:
            assert self.stru_options["lead_L"]["bloch_factor"] == self.stru_options["lead_R"]["bloch_factor"], "bloch_factor should be the same for both leads in this version"
            self.useBloch = True
            self.bloch_factor = self.stru_options["lead_L"]["bloch_factor"]
        else:
            self.useBloch = False
            self.bloch_factor = [1,1,1]

        # check the consistency of the kmesh and pbc
        assert len(self.pbc) == 3, "pbc should be a list of length 3"
        for i in range(3):
            if self.pbc[i] == False and self.stru_options["kmesh"][i] > 1:
                raise ValueError("kmesh should be 1 for non-periodic direction")
            elif self.pbc[i] == False and self.stru_options["kmesh"][i] == 0:
                self.stru_options["kmesh"][i] = 1
                log.warning(msg="kmesh should be set to 1 for non-periodic direction! Automatically Setting kmesh to 1 in direction {}.".format(i))
            elif self.pbc[i] == True and self.stru_options["kmesh"][i] == 0:
                raise ValueError("kmesh should be > 0 for periodic direction")
            
        if not any(self.pbc):
            self.kpoints,self.wk = np.array([[0,0,0]]),np.array([1.])
        else:
            self.kpoints,self.wk = kmesh_sampling_negf(self.stru_options["kmesh"], 
                                                       self.stru_options["gamma_center"],
                                                     self.stru_options["time_reversal_symmetry"])
        log.info(msg="------ k-point for NEGF -----")
        log.info(msg="Gamma Center: {0}".format(self.stru_options["gamma_center"]))
        log.info(msg="Time Reversal: {0}".format(self.stru_options["time_reversal_symmetry"]))
        log.info(msg="k-points Num: {0}".format(len(self.kpoints)))
        if len(self.wk)<10:
            log.info(msg="k-points: {0}".format(self.kpoints))
            log.info(msg="k-points weights: {0}".format(self.wk))
        log.info(msg="--------------------------------")

        self.unit = unit
        self.scf = scf
        self.block_tridiagonal = block_tridiagonal
        for lead_tag in ["lead_L", "lead_R"]:
            assert "voltage" in self.stru_options[lead_tag], f"{lead_tag} voltage should be set in stru_options"
            if self.scf:
                if lead_tag in self.poisson_options:
                    if "voltage" in self.poisson_options.get(lead_tag, {}):
                        assert self.stru_options[lead_tag]["voltage"]==self.poisson_options[lead_tag]["voltage"], f"{lead_tag} voltage should be consistent"
                    else:
                        self.poisson_options[lead_tag]["voltage"] = self.stru_options[lead_tag].get("voltage", None)
            else:
                assert self.stru_options[lead_tag]["voltage"] == 0, f"{lead_tag} voltage should be 0 in non-scf calculation"

        if AtomicData_options is None:
            from dptb.utils.argcheck import get_cutoffs_from_model_options
            # get the cutoffs from model options
            r_max, er_max, oer_max  = get_cutoffs_from_model_options(model.model_options)
            AtomicData_options = {'r_max': r_max, 'er_max': er_max, 'oer_max': oer_max}
        else:
            log.warning(msg="AtomicData_options is extracted from input file. " \
                            "This may be not consistent with the model options. " \
                            "Please be careful and check the cutoffs.")
        formatted = json.dumps(AtomicData_options, indent=4)
        indented = '\n'.join(' ' * 15 + line for line in formatted.splitlines())
        log.info("The AtomicData_options is:\n%s", indented)

        # computing the hamiltonian
        self.negf_hamiltonian = NEGFHamiltonianInit(model=model,
                                                    AtomicData_options=AtomicData_options, 
                                                    structure=structure,
                                                    block_tridiagonal=self.block_tridiagonal,
                                                    pbc_negf = self.pbc, 
                                                    stru_options=self.stru_options,
                                                    unit = self.unit, 
                                                    results_path=self.results_path,
                                                    torch_device = self.torch_device)
        with torch.no_grad():
            # if useBloch is None, structure_leads_fold,bloch_sorted_indices,bloch_R_lists = None,None,None
            struct_device, struct_leads,structure_leads_fold,bloch_sorted_indices,bloch_R_lists = \
                self.negf_hamiltonian.initialize(kpoints=self.kpoints,block_tridiagnal=self.block_tridiagonal,\
                                                 useBloch=self.useBloch,bloch_factor=self.bloch_factor,\
                                                 use_saved_HS=self.use_saved_HS, saved_HS_path=self.saved_HS_path)

        self.free_charge = {} # net charge: hole - electron
        #  Regions for Poisson equation
        ## Dirichlet region: gate and leads
        ## Dielectric region: dielectrics
        ## Doped region: doped atomic sites, usually in leads region
        self.Dirichlet_region = [self.poisson_options[i] for i in self.poisson_options if i.startswith("gate")\
                                    or i.startswith("lead")]
        self.dielectric_region = [self.poisson_options[i] for i in self.poisson_options if i.startswith("dielectric")]
        self.doped_region = [self.poisson_options[i] for i in self.poisson_options if i.startswith("doped")]



        log.info(msg="-------------Fermi level calculation-------------")
        e_fermi = {}; chemiPot = {}
        # calculate Fermi level
        if  self.e_fermi is None:        
            elec_cal = ElecStruCal(model=model,device=self.torch_device)
            nel_atom_lead = self.get_nel_atom_lead(
                                struct_leads, 
                                charge={lead_tag: self.stru_options[lead_tag].get("charge", 0) for lead_tag in ["lead_L", "lead_R"]}
                                )
            log.info(msg="Number of electrons in lead_L: {0}".format(nel_atom_lead["lead_L"]))
            log.info(msg="Number of electrons in lead_R: {0}".format(nel_atom_lead["lead_R"]))
            for lead_tag in ["lead_L", "lead_R"]:
                log.info(msg="-----Calculating Fermi level for {0}-----".format(lead_tag))
                _, e_fermi[lead_tag]  = elec_cal.get_fermi_level(data=struct_leads[lead_tag], 
                                nel_atom = nel_atom_lead[lead_tag],
                                meshgrid=self.stru_options[lead_tag]["kmesh_lead_Ef"],
                                AtomicData_options=AtomicData_options,
                                smearing_method=self.stru_options.get("e_fermi_smearing", "FD"),
                                temp=100.0)
        else:
            e_fermi["lead_L"] = self.e_fermi
            e_fermi["lead_R"] = self.e_fermi
            log.info(msg="Fermi level is set to {0} from input file".format(self.e_fermi))
        
        # calculate electrochemical potential
        for lead_tag in ["lead_L", "lead_R"]:
            chemiPot[lead_tag] = e_fermi[lead_tag] - self.stru_options[lead_tag]["voltage"]
        
        self.e_fermi = e_fermi
        self.chemiPot = chemiPot
        log.info(msg="-------------------------------------------------")
        if abs(self.chemiPot["lead_L"]-self.chemiPot["lead_R"]) > 5e-4: # non-zero bias case
            assert abs(self.stru_options["lead_L"]["voltage"]-self.stru_options["lead_R"]["voltage"]) > 5e-4, "This is a heterogeneous system, which is not supported in this version."
            if self.poisson_options["with_Dirichlet_leads"]:
                E_ref = 0.5 * (self.chemiPot["lead_L"] + self.chemiPot["lead_R"]) 
            else: # NanoTCAD style NEGF-Poisson SCF
                E_ref = self.e_fermi["lead_L"]
                # In NanoTCAD Vides, the reference energy is set to the Fermi level of the whole system. Here we set it to the Fermi level of lead_L.
                # In homogeneous case, the Fermi level of lead_L and lead_R are the same, so it does not matter.
            log.info(msg="Non-zero bias case detected.")
            # In this version, dpnegf does not support the heterogeneous case, where the Fermi level is different in the leads
            # because left-lead and right-lead Fermi level are calculated separately, which may be erroneous due to different vaccum level
        else: # zero bias case
            E_ref = self.e_fermi["lead_L"]
            log.info(msg="Zero bias case detected.")

        log.info(msg="Fermi level for lead_L: {0}".format(self.e_fermi["lead_L"]))
        log.info(msg="Fermi level for lead_R: {0}".format(self.e_fermi["lead_R"]))    
        log.info(msg="Electrochemical potential for lead_L: {0}".format(self.chemiPot["lead_L"]))
        log.info(msg="Electrochemical potential for lead_R: {0}".format(self.chemiPot["lead_R"]))        
        log.info(msg="Reference energy E_ref: {0}".format(E_ref))
        log.info(msg="=================================================\n")

        # initialize deviceprop and leadprop
        self.deviceprop = DeviceProperty(self.negf_hamiltonian, struct_device, results_path=self.results_path,
                                         efermi=self.e_fermi, chemiPot=chemiPot, E_ref=E_ref)
        self.deviceprop.set_leadLR(
                lead_L=LeadProperty(
                hamiltonian=self.negf_hamiltonian, 
                tab="lead_L", 
                structure=struct_leads["lead_L"], 
                results_path=self.results_path,
                e_T=self.ele_T,
                efermi=self.e_fermi["lead_L"], 
                voltage=self.stru_options["lead_L"]["voltage"],
                E_ref=E_ref,
                useBloch=self.useBloch,
                bloch_factor=self.bloch_factor,
                structure_leads_fold=structure_leads_fold["lead_L"],
                bloch_sorted_indice=bloch_sorted_indices["lead_L"],
                bloch_R_list=bloch_R_lists["lead_L"]
            ),
                lead_R=LeadProperty(
                hamiltonian=self.negf_hamiltonian, 
                tab="lead_R", 
                structure=struct_leads["lead_R"], 
                results_path=self.results_path, 
                e_T=self.ele_T,
                efermi=self.e_fermi["lead_R"], 
                voltage=self.stru_options["lead_R"]["voltage"],
                E_ref=E_ref,
                useBloch=self.useBloch,
                bloch_factor=self.bloch_factor,
                structure_leads_fold=structure_leads_fold["lead_R"],
                bloch_sorted_indice=bloch_sorted_indices["lead_R"],
                bloch_R_list=bloch_R_lists["lead_R"]
            )
        )


        # number of orbitals on atoms in device region
        self.device_atom_norbs = self.negf_hamiltonian.atom_norbs[self.negf_hamiltonian.device_id[0]:self.negf_hamiltonian.device_id[1]]
        left_connected_atom_mask = abs(struct_device.positions[:,2]-min(struct_device.positions[:,2]))<1e-6
        right_connected_atom_mask = abs(struct_device.positions[:,2]-max(struct_device.positions[:,2]))<1e-6

        self.left_connected_orb_mask = torch.tensor( [bool(p) for p, norb in zip(left_connected_atom_mask, self.device_atom_norbs) \
                                                      for _ in range(norb)],dtype=torch.bool)
        self.right_connected_orb_mask = torch.tensor( [bool(p) for p, norb in zip(right_connected_atom_mask, self.device_atom_norbs) \
                                                        for _ in range(norb)],dtype=torch.bool)


        # geting the output settings
        self.out_tc = out_tc
        self.out_dos = out_dos
        self.out_density = out_density
        self.out_potential = out_potential
        self.out_current = out_current
        self.out_current_nscf = out_current_nscf
        self.out_ldos = out_ldos
        self.out_lcurrent = out_lcurrent
        assert not (self.out_lcurrent and self.block_tridiagonal)
        self.out = {}
        # initialize density class
        self.density_options = density_options
        self.generate_energy_grid()
        if self.density_options["method"] == "Ozaki":
            self.density = Ozaki(R=self.density_options["R"], 
                                 M_cut=self.density_options["M_cut"], 
                                 n_gauss=self.density_options["n_gauss"])
            
        elif self.density_options["method"] == "Fiori":
            if self.density_options["integrate_way"] == "gauss":
                assert self.density_options["n_gauss"] is not None, "n_gauss should be set for Fiori method using gauss integration"
                self.density = Fiori(n_gauss=self.density_options["n_gauss"],
                                     integrate_way=self.density_options["integrate_way"],
                                     e_grid=self.uni_grid)
            elif self.density_options["integrate_way"] == "direct":
                self.density = Fiori(integrate_way=self.density_options["integrate_way"],
                                     e_grid=self.uni_grid) #calculate the density by integrating the energy window in direct way
            else:
                raise ValueError("integrate_way should be 'gauss' or 'direct' for Fiori method")
        else:
            raise ValueError



    def generate_energy_grid(self):

        # computing parameters for NEGF
        
        cal_pole = False
        cal_int_grid = False

        if self.scf:
            v_list = [self.stru_options[i].get("voltage", None) for i in self.stru_options if i.startswith("lead")]
            v_list_b = [i == v_list[0] for i in v_list]
            if not all(v_list_b):
                if self.density_options["method"] == "Ozaki": 
                    cal_pole = True
                cal_int_grid = True
        elif self.out_density or self.out_potential:
            if self.density_options["method"] == "Ozaki":
                cal_pole = True
            v_list = [self.stru_options[i].get("voltage", None) for i in self.stru_options if i.startswith("lead")]
            v_list_b = [i == v_list[0] for i in v_list]
            if not all(v_list_b):
                cal_int_grid = True

        if self.out_lcurrent:
            cal_int_grid = True
        
        if self.out_current:
            cal_int_grid = True

        if self.out_dos or self.out_tc or self.out_current_nscf or self.out_ldos:
            # Energy gird is set relative to Fermi level
            self.uni_grid = torch.linspace(start=self.emin, end=self.emax, steps=int((self.emax-self.emin)/self.espacing))

        if cal_pole and  self.density_options["method"] == "Ozaki":
            self.poles, self.residues = ozaki_residues(M_cut=self.density_options["M_cut"])
            self.poles = 1j* self.poles * self.kBT + self.deviceprop.lead_L.chemiPot - self.deviceprop.chemiPot

        if cal_int_grid:
            xl = torch.tensor(min(v_list)-8*self.kBT)
            xu = torch.tensor(max(v_list)+8*self.kBT)
            self.int_grid, self.int_weight = gauss_xw(xl=xl, xu=xu, n=int(self.density_options["n_gauss"]))

    def compute(self):

        if self.scf:

            # create real-space grid
            grid = self.get_grid(self.poisson_options["grid"],self.deviceprop.structure)

            # create Dirichlet boundary condition region
            Dirichlet_group = []
            for idx in range(len(self.Dirichlet_region)):
                Dirichlet_init = Dirichlet(self.Dirichlet_region[idx].get("x_range",None).split(':'),\
                                self.Dirichlet_region[idx].get("y_range",None).split(':'),\
                                self.Dirichlet_region[idx].get("z_range",None).split(':'))
                #TODO: when heterogenous Dirichlet conditions are set, the voltage should be set as electrochemical potential(Fermi level + voltage)
                Dirichlet_init.Ef = -1*float(self.Dirichlet_region[idx].get("voltage",None)) # in unit of eV
                Dirichlet_group.append(Dirichlet_init)
            
            # create dielectric region          
            dielectric_group = []
            for dd in range(len(self.dielectric_region)):
                dielectric_init = Dielectric(   self.dielectric_region[dd].get("x_range",None).split(':'),\
                                                self.dielectric_region[dd].get("y_range",None).split(':'),\
                                                self.dielectric_region[dd].get("z_range",None).split(':'))
                dielectric_init.eps = float(self.dielectric_region[dd].get("relative permittivity",None))
                dielectric_group.append(dielectric_init) 

            # create interface
            interface_poisson = Interface3D(grid,Dirichlet_group,dielectric_group)
            interface_poisson.get_potential_eps(Dirichlet_group+dielectric_group)
            atom_gridpoint_index =  list(interface_poisson.grid.atom_index_dict.values()) # atomic site index in the grid
            for dp in range(len(self.doped_region)):
                interface_poisson.get_fixed_charge( self.doped_region[dp].get("x_range",None).split(':'),\
                                                    self.doped_region[dp].get("y_range",None).split(':'),\
                                                    self.doped_region[dp].get("z_range",None).split(':'),\
                                                    self.doped_region[dp].get("charge",None),\
                                                    atom_gridpoint_index)

            #initial guess for electrostatic potential
            log.info(msg="-----Initial guess for electrostatic potential----")
            interface_poisson.solve_poisson_NRcycle(method=self.poisson_options['solver'],\
                                                    tolerance=self.poisson_options['tolerance'],\
                                                    dtype=self.poisson_options['poisson_dtype'])
            log.info(msg="-------------------------------------------\n")

            self.poisson_negf_scf(  interface_poisson=interface_poisson,atom_gridpoint_index=atom_gridpoint_index,\
                                    err=self.poisson_options['err'],max_iter=self.poisson_options['max_iter'],\
                                    mix_rate=self.poisson_options['mix_rate'],tolerance=self.poisson_options['tolerance'])
            # calculate transport properties with converged potential
            self.negf_compute(scf_require=False,Vbias=self.potential_at_orb)
        
        else:
            profiler = Profiler()
            profiler.start() 
            self.negf_compute(scf_require=False,Vbias=None)
            profiler.stop()
            output_path = os.path.join(self.results_path, "profile_report.html")
            with open(output_path, 'w') as report_file:
                report_file.write(profiler.output_html())

    def poisson_negf_scf(self,interface_poisson,atom_gridpoint_index,err=1e-6,max_iter=1000,
                         mix_method:str='linear', mix_rate:float=0.3, tolerance:float=1e-7,Gaussian_sigma:float=3.0):

        
        # profiler.start() 
        max_diff_phi = 1e30
        max_diff_list = [] 
        iter_count=0
        mix_method_list = ['linear', 'PDIIS', 'DIIS', 'BroydenFirst', 'BroydenSecond', 'Anderson']
        if mix_method not in mix_method_list:
            raise ValueError("mix_method should be one of {}".format(mix_method_list))
        else:
        # initialize the mixer
            log.info(msg="Using {} mixing method for NEGF-Poisson SCF".format(mix_method))
            if mix_method == 'PDIIS':
                mixer = PDIISMixer(init_x=interface_poisson.phi.copy(), mix_rate=mix_rate)
            elif mix_method == 'DIIS':
                mixer = DIISMixer(max_hist=6, alpha=0.2)
            elif mix_method == 'BroydenFirst':
                mixer = BroydenFirstMixer(init_x=interface_poisson.phi, alpha=mix_rate)
            elif mix_method == 'BroydenSecond':
                mixer = BroydenSecondMixer(shape=interface_poisson.phi.shape, max_hist=8, alpha=mix_rate)
            elif mix_method == 'Anderson':
                mixer = AndersonMixer(m=5, alpha=0.2)
            elif mix_method == 'linear':
                mixer = None

   

        # Gummel type iteration
        while max_diff_phi > err:
            # update Hamiltonian by modifying onsite energy with potential
            self.potential_at_atom = interface_poisson.phi[atom_gridpoint_index]
            self.potential_at_orb = torch.cat([torch.full((norb,), p) for p, norb\
                                                in zip(self.potential_at_atom, self.device_atom_norbs)])              
            self.negf_compute(scf_require=True,Vbias=self.potential_at_orb)
            # Vbias makes sense for orthogonal basis as in NanoTCAD
            # TODO: check if Vbias makes sense for non-orthogonal basis 
            # TODO: check the sign of free_charge
            # TODO: check the spin degenracy
            # TODO: add k summation operation
            free_charge_allk = torch.zeros_like(torch.tensor(self.device_atom_norbs))
            for ik,k in enumerate(self.kpoints):
                free_charge_allk += np.real(self.free_charge[str(k)].numpy()) * self.wk[ik]
            interface_poisson.free_charge[atom_gridpoint_index] = free_charge_allk
            

            interface_poisson.phi_old = interface_poisson.phi.copy()
            max_diff_phi = interface_poisson.solve_poisson_NRcycle(method=self.poisson_options['solver'],\
                                                                tolerance=tolerance,\
                                                                dtype=self.poisson_options['poisson_dtype'])
            if mix_method == 'linear':
                interface_poisson.phi = interface_poisson.phi + mix_rate*(interface_poisson.phi_old-interface_poisson.phi)
            elif mix_method == 'DIIS':
                residual = interface_poisson.phi - interface_poisson.phi_old
                interface_poisson.phi = mixer.update(interface_poisson.phi.copy(), residual)
            elif mix_method == 'PDIIS':
                interface_poisson.phi = mixer.update(interface_poisson.phi.copy())
            elif mix_method == 'BroydenFirst':
                residual = interface_poisson.phi - interface_poisson.phi_old
                interface_poisson.phi = mixer.update(f = residual) # fixed point problem: f defined as F(\phi)-\phi =0
            elif mix_method == 'BroydenSecond':
                residual = interface_poisson.phi - interface_poisson.phi_old
                interface_poisson.phi = mixer.update(interface_poisson.phi.copy(), residual)
            elif mix_method == 'Anderson':
                interface_poisson.phi = mixer.update(interface_poisson.phi.copy(), interface_poisson.phi_old.copy())

            iter_count += 1 # Gummel type iteration
            log.info(msg="Poisson-NEGF iteration: {}    Potential Diff Maximum: {}\n".format(iter_count,max_diff_phi))
            max_diff_list.append(max_diff_phi)

            if max_diff_phi <= err:
                log.info(msg="Poisson-NEGF SCF Converges Successfully!")
            if max_diff_phi > 1e8:
                raise RuntimeError("Poisson-NEGF iteration diverges, max_diff_phi = {}".format(max_diff_phi))
            if np.isnan(max_diff_phi):
                raise RuntimeError("Poisson-NEGF iteration diverges, max_diff_phi = {}".format(max_diff_phi))
                

            if iter_count > max_iter:
                log.warning(msg="Warning! Poisson-NEGF iteration exceeds the upper limit of iterations {}".format(int(max_iter)))
                break
                # profiler.stop()
                # with open('profile_report.html', 'w') as report_file:
                #     report_file.write(profiler.output_html())
                # break

        self.poisson_out = {}
        self.poisson_out['potential'] = torch.tensor(interface_poisson.phi)
        self.poisson_out['potential_at_atom'] = self.potential_at_atom
        self.poisson_out['grid_point_number'] = interface_poisson.grid.Np
        self.poisson_out['grid'] = torch.tensor(interface_poisson.grid.grid_coord)
        self.poisson_out['free_charge_at_atom'] = torch.tensor(interface_poisson.free_charge[atom_gridpoint_index])
        self.poisson_out['max_diff_list'] = torch.tensor(max_diff_list)
        torch.save(self.poisson_out, self.results_path+"/poisson.out.pth")



        # output the profile report in html format
        # if iter_count <= max_iter: 
        #     profiler.stop()
        #     with open('profile_report.html', 'w') as report_file:
        #         report_file.write(profiler.output_html())、

    def prepare_self_energy(self, scf_require: bool) -> None:
        """
        Prepares the self-energy for the NEGF calculation.

        Depending on the calculation settings, this method either loads previously saved self-energy data
        or computes and saves new self-energy values for the device leads. The computation method varies
        based on whether self-consistent field (SCF) calculations are required and whether Dirichlet boundary
        conditions are applied to the leads.

        Parameters:
        ----------
            scf_require (bool): Indicates whether SCF calculations are required.
        """
        # self energy calculation
        log.info(msg="------Self-energy calculation------")
        if  self.self_energy_save_path is None:
            self.self_energy_save_path = os.path.join(self.results_path, "self_energy") 
        os.makedirs(self.self_energy_save_path, exist_ok=True)

        if self.use_saved_se:
            assert _has_saved_self_energy(self.self_energy_save_path), "No saved self-energy found in {}".format(self.self_energy_save_path)
            log.info(msg="Using saved self-energy from {}".format(self.self_energy_save_path))
            log.info(msg="Ensure the saved self-energy is consistent with the current calculation setting!")
        else:
            log.info(msg="Calculating self-energy and saving to {}".format(self.self_energy_save_path))
            if scf_require and self.poisson_options["with_Dirichlet_leads"]:
                # For the Dirichlet leads, the self-energy of the leads is only calculated once and saved.
                # In each iteration, the self-energy of the leads is not updated.
                # for ik, k in enumerate(self.kpoints):
                #     for e in self.density.integrate_range:
                #         self.deviceprop.lead_L.self_energy(kpoint=k, energy=e, eta_lead=self.eta_lead, save=True)
                #         self.deviceprop.lead_R.self_energy(kpoint=k, energy=e, eta_lead=self.eta_lead, save=True)
                compute_all_self_energy(self.eta_lead, self.deviceprop.lead_L, self.deviceprop.lead_R,
                                        self.kpoints, self.density.integrate_range, self.self_energy_save_path)
            elif not self.scf:
                # In non-scf case, the self-energy of the leads is calculated for each energy point in the energy grid.
                compute_all_self_energy(self.eta_lead, self.deviceprop.lead_L, self.deviceprop.lead_R,
                                        self.kpoints, self.uni_grid, self.self_energy_save_path)
        log.info(msg="-----------------------------------\n")



    def negf_compute(self,scf_require=False,Vbias=None):
        
        assert scf_require is not None, "scf_require should be set to True or False"
        self.out['k']=[];self.out['wk']=[]
        if hasattr(self, "uni_grid"): self.out["uni_grid"] = self.uni_grid

        self.prepare_self_energy(scf_require)

        for ik, k in enumerate(self.kpoints):

            self.out['k'].append(k)
            self.out['wk'].append(self.wk[ik])
            self.free_charge.update({str(k):torch.zeros_like(torch.tensor(self.device_atom_norbs),dtype=torch.complex128)})
            log.info(msg="Properties computation at k = [{:.4f},{:.4f},{:.4f}]".format(float(k[0]),float(k[1]),float(k[2])))

            if scf_require:
                if self.density_options["method"] == "Fiori":                    
                    if not self.poisson_options["with_Dirichlet_leads"]:
                        # Follow the NanoTCAD convention for NEGF-Poisson SCF
                        # without Dirichlet leads, the voltage is set as the average of the potential at the most left and right parts
                        for ll in self.stru_options.keys():
                            if ll.startswith("lead"):
                                if Vbias is not None  and self.density_options["method"] == "Fiori":
                                    # set voltage as -1*potential_at_orb[0] and -1*potential_at_orb[-1] for self-energy same as in NanoTCAD
                                    if ll == 'lead_L' :
                                        getattr(self.deviceprop, ll).voltage = Vbias[self.left_connected_orb_mask].mean()
                                        # getattr(self.deviceprop, ll).voltage = Vbias[0]
                                    else:
                                        getattr(self.deviceprop, ll).voltage = Vbias[self.right_connected_orb_mask].mean()
                                        # getattr(self.deviceprop, ll).voltage = Vbias[-1]                                     
                    else:
                        # TODO: consider the case with heterogeneous Dirichlet leads
                        # In this case, the Dirichlet conditions in leads and gate are set as electrochemical potential(Fermi level + voltage)
                        for lead_tag in ["lead_L", "lead_R"]:
                            assert getattr(self.deviceprop, lead_tag).voltage == self.stru_options[lead_tag]["voltage"]

                    if self.negf_hamiltonian.subblocks is None:
                            self.negf_hamiltonian.subblocks = self.negf_hamiltonian.get_hs_device(only_subblocks=True)
                    
                    self.density.density_integrate_Fiori(
                        e_grid = self.uni_grid, 
                        kpoint=k,
                        Vbias=Vbias,
                        block_tridiagonal=self.block_tridiagonal,
                        subblocks=self.negf_hamiltonian.subblocks,
                        integrate_way = self.density_options["integrate_way"],  
                        deviceprop=self.deviceprop,
                        device_atom_norbs=self.device_atom_norbs,
                        potential_at_atom = self.potential_at_atom,
                        with_Dirichlet_leads = self.poisson_options["with_Dirichlet_leads"],
                        free_charge = self.free_charge,
                        eta_lead = self.eta_lead,
                        eta_device = self.eta_device,
                        E_ref = self.deviceprop.E_ref
                        )
                else:
                    # TODO: add Ozaki support for NanoTCAD-style SCF
                    raise ValueError("Ozaki method does not support Poisson-NEGF SCF in this version.")
                

            # in non-scf case, computing properties in uni_gird
            else:
                if hasattr(self, "uni_grid"):                     
                    output_freq = int(len(self.uni_grid)/10)
                    if output_freq == 0: output_freq = 1
                    for ie, e in enumerate(self.uni_grid):
                        if ie % output_freq == 0:
                            log.info(msg="computing green's function at e = {:.3f}".format(float(e)))
                        if self.scf:
                            if not self.poisson_options["with_Dirichlet_leads"]:
                                for ll in self.stru_options.keys():
                                    if ll.startswith("lead") and\
                                        Vbias is not None  and\
                                        self.density_options["method"] == "Fiori":
                                            # set voltage as -1*potential_at_orb[0] and -1*potential_at_orb[-1] for self-energy same as in NanoTCAD
                                            if ll == 'lead_L':
                                                getattr(self.deviceprop, ll).voltage = Vbias[self.left_connected_orb_mask].mean()
                                            else:
                                                getattr(self.deviceprop, ll).voltage = Vbias[self.right_connected_orb_mask].mean()
                            else:
                                # TODO: consider the case with heterogeneous Dirichlet leads
                                # In this case, the Dirichlet conditions in leads and gate are set as electrochemical potential(Fermi level + voltage)
                                assert getattr(self.deviceprop, "lead_L").voltage == self.stru_options["lead_L"]["voltage"]
                                assert getattr(self.deviceprop, "lead_R").voltage == self.stru_options["lead_R"]["voltage"]    
                            
                            for ll in self.stru_options.keys():
                                if ll.startswith("lead"):
                                    getattr(self.deviceprop, ll).self_energy(
                                        energy=e, 
                                        kpoint=k, 
                                        eta_lead=self.eta_lead,
                                        method=self.sgf_solver,
                                        save_path=self.self_energy_save_path,
                                        se_info_display=self.se_info_display
                                        )
                                    # self.out[str(ll)+"_se"][str(e.numpy())] = getattr(self.deviceprop, ll).se
                                    
                        else:
                            for ll in self.stru_options.keys():
                                if ll.startswith("lead"):
                                    getattr(self.deviceprop, ll).self_energy(
                                        energy=e, 
                                        kpoint=k, 
                                        eta_lead=self.eta_lead,
                                        method=self.sgf_solver,
                                        save_path=self.self_energy_save_path,
                                        se_info_display=self.se_info_display
                                        )                                

                        self.deviceprop.cal_green_function(
                            energy=e, kpoint=k, 
                            eta_device=self.eta_device,
                            block_tridiagonal=self.block_tridiagonal,
                            Vbias=Vbias
                            )
                        # self.out["gtrans"][str(e.numpy())] = gtrans

                        
                        if self.out_dos:
                            # prop = self.out.setdefault("DOS", [])
                            # prop.append(self.compute_DOS(k))
                            prop = self.out.setdefault('DOS', {})
                            propk = prop.setdefault(str(k), [])
                            propk.append(self.compute_DOS(k))
                        if self.out_tc or self.out_current_nscf:
                            # prop = self.out.setdefault("TC", [])
                            # prop.append(self.compute_TC(k))
                            prop = self.out.setdefault('T_k', {})
                            propk = prop.setdefault(str(k), [])
                            propk.append(self.compute_TC(k))                            
                        if self.out_ldos:
                            # prop = self.out['LDOS'].setdefault(str(k), [])
                            # prop.append(self.compute_LDOS(k))
                            prop = self.out.setdefault('LDOS', {})
                            propk = prop.setdefault(str(k), [])
                            propk.append(self.compute_LDOS(k))
                        
                            
                    # over energy loop in uni_gird
                    # The following code is for output properties before NEGF ends
                    # TODO: check following code for multiple k points calculation
                
                    if self.out_density or self.out_potential:
                        if self.density_options["method"] == "Ozaki":
                            prop_DM_eq = self.out.setdefault('DM_eq', {})
                            prop_DM_neq = self.out.setdefault('DM_neq', {})
                            prop_DM_eq[str(k)], prop_DM_neq[str(k)] = self.compute_density_Ozaki(k,Vbias)
                        elif self.density_options["method"] == "Fiori":
                            log.warning("Fiori method is under test in this version.")
                            try:
                                if self.negf_hamiltonian.subblocks is None:
                                    self.negf_hamiltonian.subblocks = \
                                        self.negf_hamiltonian.get_hs_device(only_subblocks=True)

                                self.density.density_integrate_Fiori(
                                    e_grid = self.uni_grid, 
                                    kpoint=k,
                                    Vbias=Vbias,
                                    block_tridiagonal=self.block_tridiagonal,
                                    subblocks=self.negf_hamiltonian.subblocks,
                                    integrate_way = self.density_options["integrate_way"],  
                                    deviceprop=self.deviceprop,
                                    device_atom_norbs=self.device_atom_norbs,
                                    potential_at_atom = self.potential_at_atom,
                                    free_charge = self.free_charge,
                                    eta_lead = self.eta_lead,
                                    eta_device = self.eta_device
                                    )
                                prop_freecharge = self.out.setdefault('FREE_CHARGE', {})
                                prop_freecharge[str(k)] = self.free_charge[str(k)]
                            except:
                                log.warning("Free charge output has some problems.")
                        else:
                            raise ValueError("Unknown method for density calculation.")
                    if self.out_potential:
                        pass
                    if self.out_dos:
                        self.out["DOS"][str(k)] = torch.stack(self.out["DOS"][str(k)])
                    if self.out_tc or self.out_current_nscf:
                        self.out["T_k"][str(k)] = torch.stack(self.out["T_k"][str(k)])
                    # if self.out_current_nscf:
                    #     self.out["BIAS_POTENTIAL_NSCF"], self.out["CURRENT_NSCF"] = self.compute_current_nscf(k, self.uni_grid, self.out["TC"]) 
                    # computing properties that are not functions of E (improvement can be made here in properties related to integration of energy window of fermi functions)
                    if self.out_current:
                        pass
            
                    # TODO: check the following code for multiple k points calculation
                    if self.out_lcurrent:
                        lcurrent = 0
                        log.info(msg="computing local current at k = [{:.4f},{:.4f},{:.4f}]".format(float(k[0]),float(k[1]),float(k[2])))
                        for i, e in enumerate(self.int_grid):
                            log.info(msg=" computing green's function at e = {:.3f}".format(float(e)))
                            for ll in self.stru_options.keys():
                                if ll.startswith("lead"):
                                    getattr(self.deviceprop, ll).self_energy(
                                        energy=e, 
                                        kpoint=k, 
                                        eta_lead=self.eta_lead,
                                        method=self.sgf_solver,
                                        save_path=self.self_energy_save_path,
                                        se_info_display=self.se_info_display
                                        )
                                    
                            self.deviceprop.cal_green_function(
                                energy=e,
                                kpoint=k, 
                                eta_device=self.eta_device, 
                                block_tridiagonal=self.block_tridiagonal
                                )
                            
                            lcurrent += self.int_weight[i] * self.compute_lcurrent(k)

                        prop_local_current = self.out.setdefault('LOCAL_CURRENT', {})
                        prop_local_current[str(k)] = lcurrent


        if scf_require==False:
            self.out["k"] = np.array(self.out["k"])
            self.out['T_avg'] = torch.tensor(self.out['wk']) @ torch.stack(list(self.out["T_k"].values()))
            # TODO:check the following code for multiple k points calculation
            if self.out_current_nscf:
                self.out["BIAS_POTENTIAL_NSCF"], self.out["CURRENT_NSCF"] = self.compute_current_nscf(self.uni_grid, self.out["T_avg"])
            torch.save(self.out, self.results_path+"/negf.out.pth")

                
            

    def get_grid(self,grid_info,structase):
        x_start,x_end,x_num = grid_info.get("x_range",None).split(':')
        xg = np.linspace(float(x_start),float(x_end),int(x_num))

        y_start,y_end,y_num = grid_info.get("y_range",None).split(':')
        yg = np.linspace(float(y_start),float(y_end),int(y_num))
        # yg = np.array([(float(y_start)+float(y_end))/2]) # TODO: temporary fix for 2D case

        z_start,z_end,z_num = grid_info.get("z_range",None).split(':')
        zg = np.linspace(float(z_start),float(z_end),int(z_num))

        device_atom_coords = structase.get_positions()
        xa,ya,za = device_atom_coords[:,0],device_atom_coords[:,1],device_atom_coords[:,2]

        # grid = Grid(xg,yg,zg,xa,ya,za)
        grid = Grid(xg,yg,za,xa,ya,za) #TODO: change back to zg
        return grid     

    def get_nel_atom_lead(self, struct_leads, charge:float=None):
        nel_atom = self.stru_options.get("nel_atom", None)
        if nel_atom is None:
            log.warning(msg="nel_atom is None, using valence electron number by default")
        nel_atom_lead = {}
        for lead_tag in ["lead_L", "lead_R"]:
            nel_atom_lead[lead_tag] = {}
            unique_elements = struct_leads[lead_tag].get_chemical_symbols()
            for elem in unique_elements:
                if nel_atom is None:
                    if elem not in valence_electron:
                        raise ValueError(f"Element {elem} is not in the valence electron dictionary")
                    nel_atom_lead[lead_tag][elem] = valence_electron[elem]
                else:
                    if elem not in nel_atom:
                        raise ValueError(f"Element {elem} is not in the nel_atom dictionary")
                    nel_atom_lead[lead_tag][elem] = nel_atom[elem]
            # subtract dope charge if the lead is doped
            if charge is not None:
                assert charge.get(lead_tag) is not None, f"Charge for {lead_tag} is not provided"
                if isinstance(charge[lead_tag], (int, float)):
                    if charge[lead_tag] < 0:
                        log.info(msg=f"p doping detected in {lead_tag}, fixed_charge = {charge[lead_tag]}")
                    elif charge[lead_tag] > 0:
                        log.info(msg=f"n doping detected in {lead_tag}, fixed_charge = {charge[lead_tag]}")
                    else:
                        log.warning(msg=f"No doping detected in {lead_tag}, fixed_charge = {charge[lead_tag]}")
                else:
                    raise ValueError(f"Charge for {lead_tag} should be a number, got {type(charge[lead_tag])}")
                nel_atom_lead[lead_tag] = {elem: nel_atom_lead[lead_tag][elem] + charge[lead_tag] for elem in nel_atom_lead[lead_tag]}

        return nel_atom_lead  
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp(x / self.kBT))
    
    def compute_properties(self, kpoint, properties):
        
        # for k in self.kpoints:
        #     ik = update_kmap(self.results_path, kpoint=k)
        for p in properties:
            # log.info(msg="Computing {0} at k = {1}".format(p, k))
            prop: list = self.out.setdefault(p, [])
            prop.append(getattr(self, "compute_"+p)(kpoint))


    def compute_DOS(self, kpoint):
        return self.deviceprop.dos
    
    def compute_TC(self, kpoint):
        return self.deviceprop.tc
    
    def compute_LDOS(self, kpoint):
        return self.deviceprop.ldos
    
    def compute_current_nscf(self, ee, tc):
        return self.deviceprop._cal_current_nscf_(ee, tc)

    def compute_density_Ozaki(self, kpoint,Vbias):
        DM_eq, DM_neq = self.density.integrate(deviceprop=self.deviceprop, kpoint=kpoint, Vbias=Vbias, block_tridiagonal=self.block_tridiagonal)
        return DM_eq, DM_neq
     

    def compute_current(self, kpoint):
        self.deviceprop.cal_green_function(e=self.int_grid, kpoint=kpoint, block_tridiagonal=self.block_tridiagonal)
        return self.deviceprop.current
    
    def compute_lcurrent(self, kpoint):
        return self.deviceprop.lcurrent


    def SCF(self):
        pass

