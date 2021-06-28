import torch
import numpy as np
from params_pytorch_sp import get_hop_int as get_hop_int_sp
from params_pytorch_spd import get_hop_int as get_hop_int_spd

from typing import Callable, Union, Tuple, Any
from functorch import vmap

Tensor = torch.tensor
from torch import nn
import xitorch
from xitorch import linalg as xilinalg
import pymatgen
from pymatgen.io.vasp import outputs
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
import pandas as pd
from skfio.problem import Problem

import matplotlib.pyplot as plt


torch.set_default_dtype(torch.double)
torch.set_printoptions(7)


def eval_vasp_xml(file="vasprun.xml", recip=False, norm_fermi=True, print_out=False, spin_up=True, slice_size=1):
    dft = pymatgen.io.vasp.outputs.Vasprun(file, parse_projected_eigen=False)
    orbital_energy = pd.read_csv("element_orbital_energy.csv").set_index("element")

    lattice = torch.tensor(dft.get_trajectory().as_dict()['lattice']).squeeze()
    lattice_normed = lattice / torch.linalg.norm(lattice, axis=1, keepdims=True)
    lattice_recip = torch.tensor(Lattice(lattice).reciprocal_lattice.matrix)  # wrong!

    positions_base = torch.tensor(dft.get_trajectory().as_dict()['base_positions'])
    positions = torch.matmul(positions_base, lattice)  # dot?

    k_points = torch.tensor(dft.actual_kpoints)

    weights = torch.tensor(dft.actual_kpoints_weights)  # how to use ?

    species_dict = {}
    species_arr = np.asarray(dft.atomic_symbols)
    count = 0
    for key in dict.fromkeys(set(dft.atomic_symbols), {}):
        species_dict["species_" + Element(key).long_name] = {"symbol": key,
                                                             "number": count,
                                                             "Es": orbital_energy.loc[key, "E_s"],
                                                             "Ep": orbital_energy.loc[key, "E_p"],
                                                             "Ed": orbital_energy.loc[key, "E_d"],
                                                             }
        species_arr[species_arr == key] = count  # cycles through elements but returns correct one anyway
        count += 1

    species_arr = torch.tensor(species_arr.astype(int))

    for key in dft.eigenvalues.keys():
        key_last = key
    true = torch.zeros(
        (dft.eigenvalues[key_last][:, :, 0].shape[0], dft.eigenvalues[key_last][:, :, 0].shape[1],
         len(dft.eigenvalues.keys())))
    count = 0
    if len(dft.eigenvalues.keys()) != 1:
        print("only one spin direction supported but", len(dft.eigenvalues.keys()), "where given")
    for key in dft.eigenvalues.keys():  # OrderedDictionary might be nice
        true[:, :, count] = torch.tensor(dft.eigenvalues[key][:, :, 0])  # what is [:, :, 0] ???????????????????

        # print("dft.eigenvalues[key][:, :, 1]", dft.eigenvalues[key][:, :, 1])
        occupied = torch.max(torch.tensor(np.nonzero(dft.eigenvalues[key][:, :, 1])[1]) + 1)
        fermi = find_fermi(true, occupied)
        print("highest: ", occupied)
        count += 1
    if norm_fermi:
        true -= fermi
        print("E fermi calculated normed", find_fermi(true, occupied))
    true_up = true[:, :, 0]  # for one spin type in system
    if slice_size > 1:
        print("Kpts reduced from", k_points.shape, "to:", k_points[0:k_points.shape[0]:slice_size, :].shape)
        k_points = k_points[0:k_points.shape[0]:slice_size, :]
        true = true[0:true.shape[0]:slice_size, :, :]
        true_up = true_up[0:true_up.shape[0]:slice_size, :]
    if spin_up:
        true_return = true_up
    else:
        true_return = true
    if print_out:
        print("Lattice", type(lattice), lattice.shape, "\n", lattice)
        print("Lattice Normed", type(lattice_normed), lattice_normed.shape, lattice_normed)
        print("Lattice recip", type(lattice_recip), lattice_recip.shape, "\n", lattice_recip)
        print("Positions", type(positions_base), positions_base.shape, positions_base)
        print("Positions dot", type(positions), positions.shape, "\n", positions)
        print("kpts", k_points.shape, k_points)
        print("weights", weights.shape, weights)
        print("True shape", true_return.shape, true)
        print("species", species_arr.shape, species_arr, "\n", species_dict)
        # print("true", dft.eigenvalues[:].shape, "\n", dft.eigenvalues[dft.eigenvalues.keys()[0]][0, :, 0], "\n",
        #       dft.eigenvalues[dft.eigenvalues.keys()[0]][0, :, 1])
        print("E fermi vasp", dft.efermi)
        print("Highest occupied", occupied)
        print("E fermi calculated", fermi)
    if recip:
        return k_points, weights, lattice_recip, positions, species_arr, species_dict, true_return, occupied
    else:
        return k_points, weights, lattice, positions, species_arr, species_dict, true_return, occupied



def eval_model(species_dict, model_poly=12, model_gauss=3, print_out=False, gauss=False, spd=False):
    # model = "P_12"  # "P_15 * G_3"  # poly should be even
    if gauss:
        model = "P_" + str(model_poly) + " * G_" + str(model_gauss)
    else:
        model = "P_" + str(model_poly)
    species_dict["models"] = {
        "global": True,
        "model": model
    }
    pr = Problem.fromDictionary(**species_dict)

    modelData = pr.makeModelData(model, ir0=19)
    # print("Modeldata", modelData.keys())

    species_count = len(species_dict) - 1
    orbital_energy = np.zeros((species_count, 3))

    # Get Params from SK-file with model
    params_tb = {}
    params_overlap = {}
    key_list = []
    if spd:
        key_list = reversed(sorted(modelData.keys()))
        print(key_list)
    else:
        for key in modelData:
            if not "d" in key:
                key_list.append(key)
    for key in key_list:
        if not gauss:
            if key[0] == "V":
                params[key] = torch.tensor(modelData[key])
            elif key[0] == "S":
                params_overlap[key] = torch.tensor(modelData[key])
        else:
            if key[0] == "V":  # and "d" in key
                params_tb[key] = {'gauss': {}}
                params = torch.tensor(modelData[key])
                params_tb[key]['polynom'] = params[:, :, :model_poly].view(species_count, species_count, model_poly)
                n = (params.shape[-1]-model_poly) // 3
                params_tb[key]['gauss']['cc'] = params[:, :, model_poly:model_poly+n].view(species_count, species_count, n)
                params_tb[key]['gauss']['aa'] = params[:, :, model_poly+n:model_poly+2*n].view(species_count, species_count, n)
                params_tb[key]['gauss']['x0'] = params[:, :, model_poly+2*n:].view(species_count, species_count, n)
            elif key[0] == "S":
                params_overlap[key] = {'gauss': {}}
                params = torch.tensor(modelData[key])
                params_overlap[key]['polynom'] = params[:, :, :model_poly].view(species_count, species_count, model_poly)
                n = (params.shape[-1]-model_poly) // 3
                params_overlap[key]['gauss']['cc'] = params[:, :, model_poly:model_poly+n].view(species_count, species_count, n)
                params_overlap[key]['gauss']['aa'] = params[:, :, model_poly+n:model_poly+2*n].view(species_count, species_count, n)
                params_overlap[key]['gauss']['x0'] = params[:, :, model_poly+2*n:].view(species_count, species_count, n)
    # needs optimization !!!!!!!!!!
    if spd:
        params_tb_over = {'V': {
                            'Vsss': params_tb['Vsss'],
                            'Vsps': params_tb['Vsps'],
                            'Vpps': params_tb['Vpps'],
                            'Vppp': params_tb['Vppp'],
                            'Vsds': params_tb['Vsds'],
                            'Vpds': params_tb['Vpds'],
                            'Vpdp': params_tb['Vpdp'],
                            'Vdds': params_tb['Vdds'],
                            'Vddp': params_tb['Vddp'],
                            'Vddd': params_tb['Vddd']
                                },
                            'S': {
                             'Ssss': params_overlap['Ssss'],
                             'Ssps': params_overlap['Ssps'],
                             'Spps': params_overlap['Spps'],
                             'Sppp': params_overlap['Sppp'],
                             'Ssds': params_overlap['Ssds'],
                             'Spds': params_overlap['Spds'],
                             'Spdp': params_overlap['Spdp'],
                             'Sdds': params_overlap['Sdds'],
                             'Sddp': params_overlap['Sddp'],
                             'Sddd': params_overlap['Sddd']
                             }}
    else:
        params_tb_over = {'V': {
                            'Vsss': params_tb['Vsss'],
                            'Vsps': params_tb['Vsps'],
                            'Vpps': params_tb['Vpps'],
                            'Vppp': params_tb['Vppp']
                                },
                            'S': {
                             'Ssss': params_overlap['Ssss'],
                             'Ssps': params_overlap['Ssps'],
                             'Spps': params_overlap['Spps'],
                             'Sppp': params_overlap['Sppp']
                             }}

    for key in species_dict:
        if key != "models":
            orbital_energy[species_dict[key]["number"], 0] = species_dict[key]["Es"]
            orbital_energy[species_dict[key]["number"], 1] = species_dict[key]["Ep"]
            orbital_energy[species_dict[key]["number"], 2] = species_dict[key]["Ed"]

    params_diag = {'e_s': torch.tensor(orbital_energy[:, 0].reshape((species_count, 1))),
                   'e_px': torch.tensor(orbital_energy[:, 1].reshape((species_count, 1))),
                   'e_py': torch.tensor(orbital_energy[:, 1].reshape((species_count, 1))),
                   'e_pz': torch.tensor(orbital_energy[:, 1].reshape((species_count, 1)))
                       }
    if spd:
        params_diag.update({
                    'e_dxy': torch.tensor(orbital_energy[:, 2].reshape((species_count, 1))),
                    'e_dxz': torch.tensor(orbital_energy[:, 2].reshape((species_count, 1))),
                    'e_dyz': torch.tensor(orbital_energy[:, 2].reshape((species_count, 1))),
                    'e_dz2': torch.tensor(orbital_energy[:, 2].reshape((species_count, 1))),
                    'e_dx2-y2': torch.tensor(orbital_energy[:, 2].reshape((species_count, 1)))
                    # 'e_S': jnp.zeros((species_count)).reshape((species_count, 1))
                        })
    if print_out:
        print("Model", model)
        print("params", params)
        print("params overlap", params_overlap)
        print("params diag", params_diag)
    return params_tb_over, params_diag


def map_product(distance):
    """ vmap is used to effectively calculate the distances of the cartesian product of the particles
    Args:
        distance: distance_fn that accepts ((N1,N2,dim), (N3,dim)) arrays as input

    Returns: map prduct of distance
    """
    return vmap(vmap(vmap(distance, (0, None), 0), (1, None), 1), (None, 0), 0)


def get_dist_vec(r_a, r_b):
    return r_a - r_b


def bondmatrix_masking(cutoff):
    """Returns function that returns masking matrix that is 1 where
    distance smaller cutoff, 0 when larger cutoff
    Args:
        cutoff: distance Float
    Returns:
        Callable that returns  masking matrix from a distance matrix
    """

    # @jax.profiler.trace_function
    # @jit
    def create_bondmatrix(dr):
        mask = torch.where(torch.logical_and(dr > 0.1, dr < cutoff), 1.0, 0.0)
        return mask

    return create_bondmatrix  # mask_cutoff


def get_dir_cos(dist_vec):
    """ Calculates directional cosines from distance vectors.
    Calculate directional cosines with respect to the standard cartesian
    axes and avoid division by zero
    Args:
        dist_vec: distance vector between particles
    Returns: dir_cos, array of directional cosines of distances between particles
    """
    norm = torch.linalg.norm(dist_vec, axis=-1)
    dir_cos = dist_vec * torch.repeat_interleave(torch.unsqueeze(torch.where(
                                                                    torch.linalg.norm(dist_vec, axis=-1) == 0,
                                                                    torch.zeros(norm.shape, device=dist_vec.device),
                                                                    1 / norm), axis=-1), 3, dim=-1)
    return dir_cos


def shift_fn(r_a, shifts):
    """
    Args:
        r_a: position of particle a as vector
        shifts: uses 2D shifts matrix of comupte_shifts function
    Returns:
    """
    return r_a.view(1, shifts.shape[-1]) + shifts


def calc_phase_matrix(kpt, shift, lattice):
    """ Calculates the phases arising from a set of k-points and shifts
    Args:
        kpt: Coordinates of k-point as matrix (N_k,dim)
        shift: Matrix of shifts returned from compute_shifts function (N_shifts, dim)
        lattice: Lattice vectors as 2D matrix e.g.: jnp.array([[1.0, 0.0, 0], [0.5, jnp.sqrt(3.0)/2.0, 0], [0, 0, 10]])
    Returns:
        g_mat:
    """
    rec_lat = torch.linalg.inv(lattice)
    kpt_cart = torch.matmul(rec_lat, kpt)
    g_mat = torch.exp(2. * np.pi * 1j * torch.dot(kpt_cart, shift))
    return g_mat


def set_diagonal_to_inf(hamiltonian, value=10e9):
    """ Zero column, row pairs correspond to non-existing particles and orbitals,
        in order to separate them from the actual eigenvalues the diagonals
         of these indices will be set to a high value
    Args:
        hamiltonian: Matrix of hamiltonian (N_k, particle number*N_orbitals, particle number*N_orbitals)
        value: int/ float value the zeros are set to
    Returns:
        Hamiltonian with high eigenvalues for non existing particles/orbitals
    """
    # diag = torch.sum(torch.abs(hamiltonian), axis=0)
    # diag = torch.where(diag == 0, value, 0.)
    ### USE NEW ONE ????? ###
    diag = torch.sum(torch.abs(hamiltonian), axis=0)
    b = torch.ones_like(diag)*torch.tensor(value, device=diag.device)
    c = torch.zeros_like(diag)
    diag = torch.where(diag == torch.zeros_like(diag), b, c)
    return torch.diag(diag)


# interactions[species_a, species_b]

class DFTB(nn.Module):
    def __init__(self, n_species, cutoff, lattice, diag_params, highest_occupied, init_param_values=None, gauss=False, spd=False):
        super(DFTB, self).__init__()
        if not gauss:
            if init_param_values is None:
                self.dftb_params = nn.ModuleDict({'V': nn.ModuleDict({'Vsss': SKparam(n_species)})
                                              })
                self.dftb_params = nn.ModuleDict({'V': nn.ModuleDict({'Vsss': SKparam(n_species)})
                                              })
            else:
                self.dftb_params = nn.ModuleDict({})
                for key in init_param_values:
                    self.dftb_params[key] = nn.ModuleDict({})
                    for key2 in init_param_values[key]:
                        self.dftb_params[key][key2] = SKparam(n_species, init_param_values[key][key2])
        else:
            if init_param_values is None:
                self.dftb_params = nn.ModuleDict({'V': nn.ModuleDict({'Vsss': SKparamgauss(n_species)})
                                                  })
                self.dftb_params = nn.ModuleDict({'V': nn.ModuleDict({'Vsss': SKparamgauss(n_species)})
                                                  })
            else:
                self.dftb_params = nn.ModuleDict({})
                for key in init_param_values:
                    self.dftb_params[key] = nn.ModuleDict({})
                    for key2 in init_param_values[key]:
                        self.dftb_params[key][key2] = SKparamgauss(n_species, init_param_values[key][key2])
        """ , 'Vsps':SKparam(species),
                                          'Vpps':SKparam(species), 'Vppp':SKparam(species), 'Vsds':SKparam(species),
                                          'Vpds':SKparam(species), 'Vpdp':SKparam(species), 'Vdds':SKparam(species),
                                          'Vddp':SKparam(species), 'Vddd'}) """
        self.species_count = n_species
        self.cutoff = cutoff
        self.lattice = lattice
        self.shifts = self.compute_shift()
        self.diag_params = diag_params
        self.highest_occupied = highest_occupied
        self.spd = spd

    def compute_shift(self):
       """ calculate neccessary repetitions in each direction with reciprocal lattice
       Args:
           lattice: takes lattice matrix as 2D-Array , e.g.: np.diag(np.ones(3))
           cutoff: cutoff distance as float
       Returns:
           shifts matrix as 2D matrix of shift vectors
       """
       n_repeat = torch.ceil(torch.linalg.norm(torch.linalg.inv(self.lattice), axis=0) * self.cutoff).type(torch.int64)
       # n_repeat = np.int32(np.asarray([1, 1, 1]))
       relative_shifts = torch.tensor([[el, el2, el3] for el in range(-n_repeat[0], n_repeat[0] + 1, 1)
                                       for el2 in range(-n_repeat[1], n_repeat[1] + 1, 1)
                                       for el3 in range(-n_repeat[2], n_repeat[2] + 1, 1)])
       relative_shifts2 = torch.where(torch.where(relative_shifts > 0, relative_shifts - 1, relative_shifts) < 0,
                                      relative_shifts + 1,
                                      torch.where(relative_shifts > 0, relative_shifts - 1, relative_shifts))
       shifts = torch.matmul(
           torch.unsqueeze(self.lattice.T, axis=0).repeat_interleave(relative_shifts2.shape[0], dim=0),
           torch.unsqueeze(relative_shifts2, -1).type(self.lattice.type())).squeeze()
       relative_shifts = relative_shifts[torch.where(torch.linalg.norm(shifts, axis=1) < self.cutoff)]
       shifts = torch.matmul(
           torch.unsqueeze(self.lattice.T, axis=0).repeat_interleave(relative_shifts.shape[0], dim=0),
           torch.unsqueeze(relative_shifts, -1).type(self.lattice.type())).squeeze()
       return shifts


    def get_params(self, dr, species_a, species_b, key):
        """
        Args:
            dr: 2D matrix of distances of particles
            species_a: element of first particle
            species_b: element of second particle
            kwargs: Dict of params for SK e.g, {"V_sss":0.2, ...}
        Returns: param_vec: slayter-koster parameters for given particles and distances
        """
        # (N_particle, N_particle, N_images, 1 -> N-parameters through broadcasting)
        param_vec = torch.unsqueeze(torch.zeros(dr.shape), axis=-1)
        for i in range(self.species_count):
            for j in range(self.species_count):
                # print((i,j))
                mask_a = torch.where(species_a == i, 1.0, 0.0)
                mask_b = torch.where(species_b == j, 1.0, 0.0)
                mask = mask_a * mask_b
                param_vec = param_vec + (self.calc_tb_params(dr, key, (i, j)) * torch.unsqueeze(mask, axis=-1))  # shapes are brodcasted
                # print(param_vec[:, :, 13][:, :, 0], dr[:, :, 13], species_a[:, :, 13], species_b[:, :, 13], species_a.shape)
        return param_vec

    def calc_tb_params(self, dr, key, species_tuple):
        """Select parameters for species pair from kwargs dictionary and set onsite terms to zero
        Args:
            dr: 2D matrix of distances of particles
            cutoff: float for cutoff distance in Angstrom
            kwargs: Dict of 2D matrix of slyer-koster parameters
        Returns: parameters for sk calculation
        """
        param = []
        for SKP in self.dftb_params[key]:
            param.append(torch.where(torch.logical_or(dr <= 0.1, dr > self.cutoff), torch.zeros_like(dr),
                                     self.dftb_params[key][SKP](dr, species_tuple)))
        param = torch.stack(param, axis=-1)
        return param

    def get_params_diag(self, dr, species_a, species_b):
        """
        Args:
            dr: 2D matrix of distances of particles
            species_a: element of first particle
            species_b: element of second particle
            kwargs: Dict of params for SK e.g, {"V_sss":0.2, ...}
        Returns: param_vec: slayter-koster parameters for given particles and distances
        """
        # (N_particle, N_particle, N_images, 1 -> N-parameters through broadcasting)
        param_diag_vec = torch.unsqueeze(torch.zeros(dr.shape), axis=-1)

        for i in range(self.species_count):
            mask = torch.where(species_a == i, 1.0, 0.0)
            param_diag_vec = param_diag_vec + (
                    self.calc_diag_params(dr, i) * torch.unsqueeze(mask, axis=-1))  # shapes are brodcasted
        param_diag_vec = vmap(vmap(vmap(torch.diag, 0), 0), 0)(param_diag_vec)
        return param_diag_vec

    def calc_diag_params(self, dr, species):
        """Select parameters for species pair from kwargs dictionary and set onsite terms to zero
        Args:
            dr: 2D matrix of distances of particles
            cutoff: float for cutoff distance in Angstrom
            kwargs: Dict of 2D matrix of slyer-koster parameters
        Returns: parameters for sk calculation
        """
        param = torch.stack([torch.where(dr != 0.0, 0.0, self.diag_params[key][species])
                             for key in self.diag_params], axis=-1)
        return param

    def create_hamiltonian_wo_k(self, positions, species):
        """
        Args:
            positions: particle position matrix 2D
            species: array of species
            shifts: uses 2D shifts matrix of comupte_shifts function
            kwargs: Dict of 2D matrix of slyer-koster parameters
            kwargs_diag: Dict of 2D matrix of one-site parameters
            kwargs_overlap: Dict of 2D matrix of off-site overlap parameters
        Returns:
            hamiltonian wo k as matrix
        """
        if self.spd:
            n_orbitals = 9
            get_hop_int = get_hop_int_spd
        else:
            n_orbitals = 4
            get_hop_int = get_hop_int_sp
        # n_orbitals = 9  # 4 for sp, 9 for spd
        create_bondmatrix_mask = bondmatrix_masking(self.cutoff)
        shifted_positions = vmap(shift_fn, (0, None))(positions, self.shifts)
        shifted_pair_distance_vectors = shifted_positions.view(positions.shape[0], 1, self.shifts.shape[0], 3) -\
        positions.view(1, positions.shape[0], 1, 3)

        # expand species shape to be the same as the shifted coordinates
        # DEVICE ????????????????
        shifted_species = torch.repeat_interleave(torch.unsqueeze(species, axis=0), self.shifts.shape[0], dim=0).T
        # flatten first dimension for cartesian product
        species_b = torch.repeat_interleave(torch.unsqueeze(shifted_species, axis=1), species.shape[0], dim=1)
        species_a = torch.repeat_interleave(torch.unsqueeze(shifted_species, axis=1), species.shape[0], dim=1).permute(1, 0, 2)
        # separate into two vectors for particle pairs a,b and reshape to (particle number, particle_number, N_images)


        # DEVICE ????????????????
        dir_cos = get_dir_cos(shifted_pair_distance_vectors)  # (particle number, particle number, N_images, 3)
        pair_distances = torch.linalg.norm(shifted_pair_distance_vectors,
                                           axis=-1)  # (particle number, particle number, N_images, dim)
        bondmatrix = create_bondmatrix_mask(pair_distances)
        param_vec_1 = self.get_params(pair_distances, species_a, species_b, 'V')
        param_vec_2 = self.get_params(pair_distances, species_b, species_a, 'V')
        param_vec_1 = param_vec_1 * torch.unsqueeze(bondmatrix, axis=-1)
        param_vec_2 = param_vec_2 * torch.unsqueeze(bondmatrix, axis=-1)
        hamiltonian = get_hop_int(torch.cat([param_vec_1, dir_cos, param_vec_2], axis=-1)).permute(2, 3, 4, 0, 1)
        param_diag = self.get_params_diag(pair_distances, species_a, 'diag')
        hamiltonian = hamiltonian + param_diag

        overlap_vec_1 = self.get_params(pair_distances, species_a, species_b, 'S')
        overlap_vec_2 = self.get_params(pair_distances, species_b, species_a, 'S')
        overlap_vec_1 = overlap_vec_1 * torch.unsqueeze(bondmatrix, axis=-1)
        overlap_vec_2 = overlap_vec_2 * torch.unsqueeze(bondmatrix, axis=-1)

        overlap_matrix = get_hop_int(torch.cat([overlap_vec_1, dir_cos, overlap_vec_2], axis=-1)).permute(2, 3, 4, 0, 1)
        hamiltonian = torch.permute(hamiltonian, (0, 3, 1, 4, 2)) \
            .reshape(species.shape[0] * n_orbitals, species.shape[0] * n_orbitals, self.shifts.shape[0])
        overlap_matrix = torch.permute(overlap_matrix, (0, 3, 1, 4, 2)) \
            .reshape(species.shape[0] * n_orbitals, species.shape[0] * n_orbitals, self.shifts.shape[0])
        return hamiltonian, overlap_matrix

    def get_ham(self, ham_wo_k, kpt):
        """ calculates hamiltonian for different k-points
        Args:
            ham_wo_k: hamiltonian matrix [N_p*N_o, N_p*N_o, N_images]
            kpt: Array of coordinates of the k-points, e.g. for gamma: jnp.array([[0, 0, 0]])
            shifts: uses 2D shifts matrix of compute_shifts function
            lattice: Lattice vector as 2D matrix e.g.: jnp.array([[1.0, 0.0, 0], [0.5, jnp.sqrt(3.0)/2.0, 0], [0, 0, 10]])
        Returns:
            Hamiltonian for all k-points (N_k, N_p*N_o, N_p*N_o)
        """
        phase_matrix = vmap(vmap(calc_phase_matrix, (None, 0, None)), (0, None, None))(kpt, self.shifts, self.lattice)  # (N_k, N_images)

        # expand both matrices for automatic broadcasting # (N_k, 1,1, N_images)
        g_mat = torch.unsqueeze(torch.unsqueeze(phase_matrix, axis=1), axis=1)
        ham_wo_k = torch.unsqueeze(ham_wo_k, axis=0)  # (1,particle number*N_orbitals, particle number*N_orbitals, N_images)
        hamiltonian = ham_wo_k * g_mat  # (N_k, particle number*N_orbitals, particle number*N_orbitals, N_images)
        hamiltonian = torch.sum(hamiltonian, axis=-1)  # (N_k, particle number*N_orbitals, particle number*N_orbitals)
        hamiltonian = torch.where(torch.abs(hamiltonian) < 1e-10, torch.zeros(hamiltonian.shape, dtype=torch.cdouble),
                                  hamiltonian)
        # hamiltonian += vmap(set_diagonal_to_inf, 0)(hamiltonian)  # in calculation function
        return hamiltonian

    def forward(self, positions, species, kpts):
        ham_wo_k, overlap_wo_k = self.create_hamiltonian_wo_k(positions, species)
        hamiltonian = self.get_ham(ham_wo_k, kpts)
        hamiltonian = hamiltonian + vmap(set_diagonal_to_inf, 0)(hamiltonian)
        overlap_matrix = self.get_ham(overlap_wo_k, kpts)
        overlap_matrix = overlap_matrix + torch.unsqueeze(torch.diag(torch.ones(overlap_matrix.shape[1])), 0)
        hamiltonian = torch.where(torch.abs(hamiltonian) < 10e-10,
                                  torch.zeros(hamiltonian.shape, dtype=torch.cdouble),
                                  hamiltonian)  # useless?
        overlap_matrix = torch.where(torch.abs(overlap_matrix) < 10e-10,
                                     torch.zeros(overlap_matrix.shape, dtype=torch.cdouble), overlap_matrix)
        overlap_eig_val = []
        overlap_eig_vec = []
        for i in range(overlap_matrix.shape[0]):  # list comprehension or tensor?
            overlap_op = xitorch.LinearOperator.m(overlap_matrix[i])
            overlap_eig_single = xilinalg.symeig(overlap_op)
            overlap_eig_val.append(overlap_eig_single[0])
            overlap_eig_vec.append(overlap_eig_single[1])
        overlap_eig = (torch.stack(overlap_eig_val), torch.stack(overlap_eig_vec))
        v_div_sqrt_eig = (overlap_eig[1] * 1 / torch.sqrt(overlap_eig[0]).unsqueeze(dim=-1)).permute(0, 2, 1)
        overlap_root_inv = vmap(torch.matmul, 0)(v_div_sqrt_eig, overlap_eig[1].permute(0, 2, 1).conj())
        new_ham = torch.matmul(overlap_root_inv, torch.matmul(hamiltonian, overlap_root_inv.permute(0, 2, 1).conj())) #vmap(torch.matmul, 0, 0)(overlap_inverse, hamiltonian)
        new_ham = torch.where(torch.abs(new_ham) < 10e-10, torch.zeros(new_ham.shape, dtype=torch.cdouble),
                                  new_ham)  # useless?
        solution = []
        for i in range(new_ham.shape[0]):
            hamiltonian_op = xitorch.LinearOperator.m(new_ham[i])
            solution.append(xilinalg.symeig(hamiltonian_op)[0])
        solution = torch.stack(solution)
        solution = solution - find_fermi(solution, self.highest_occupied)
        solution = solution * 27.211396  # conversion au (atomic unit) to eV
        return solution  # solution_corrected


class SKparam(nn.Module):
    def __init__(self, species, init_values=None, n_polynom_params=12):
        super(SKparam, self).__init__()
        if init_values == None:
            self.param_matrix = nn.Parameter(torch.rand(species, species, n_polynom_params), requires_grad=True)
        else:
            self.param_matrix = nn.Parameter(init_values, requires_grad=True)
        self.bohr_au_conversion = 1.8897261258369282
    def forward(self, dr, species_tuple):
        return polyval(self.param_matrix[species_tuple], dr * self.bohr_au_conversion)


class SKparamgauss(nn.Module):
    def __init__(self, species, init_values=None, n_polynom_params=12):
        super(SKparamgauss, self).__init__()
        if init_values == None:
            self.param_matrix = nn.Parameter(torch.rand(species, species, n_polynom_params), requires_grad=True)
        else:
            self.param_matrix_pol = nn.Parameter(init_values['polynom'], requires_grad=True)
            self.param_matrix_gauss = nn.ParameterDict({})
            for key, val in init_values['gauss'].items():
                self.param_matrix_gauss[key] = nn.Parameter(val, requires_grad=True)
        self.bohr_au_conversion = 1.8897261258369282
    def forward(self, dr, species_tuple):
        gauss = self.param_matrix_gauss['cc'][species_tuple].view(1, 1, 1, -1) *\
                torch.exp(-self.param_matrix_gauss['aa'][species_tuple].view(1, 1, 1, -1)**2
                * (dr.unsqueeze(dim=-1) * self.bohr_au_conversion - self.param_matrix_gauss['x0'][species_tuple].view(1, 1, 1, -1))**2)
        return polyval(self.param_matrix_pol[species_tuple], dr * self.bohr_au_conversion) * torch.sum(gauss, dim=-1)


class SKparamExp(nn.Module):
    def __init__(self, species, init_values=None, n_polynom_params=12):
        super(SKparamExp, self).__init__()
        if init_values == None:
            self.param_matrix = nn.Parameter(torch.rand(species, species, n_polynom_params), requires_grad=True)
        else:
            self.param_matrix_pol = nn.Parameter(init_values['polynom'], requires_grad=True)
            self.param_matrix_gauss = nn.ParameterDict({})
            for key, val in init_values['gauss'].items():
                self.param_matrix_gauss[key] = nn.Parameter(val, requires_grad=True)
        self.bohr_au_conversion = 1.8897261258369282

    def forward(self, dr, species_tuple):
        gauss = self.param_matrix_gauss['cc'][species_tuple].view(1, 1, 1, -1) * \
                torch.exp(-self.param_matrix_gauss['aa'][species_tuple].view(1, 1, 1, -1) ** 2
                          * (dr.unsqueeze(dim=-1) * self.bohr_au_conversion - self.param_matrix_gauss['x0'][
                    species_tuple].view(1, 1, 1, -1)) ** 2)
        # print(torch.isnan(gauss).any())
        # print(gauss.shape, torch.sum(gauss, dim=-1).shape, self.param_matrix['gauss']['cc'][species_tuple].shape)
        return polyval(self.param_matrix_pol[species_tuple], dr * self.bohr_au_conversion) * torch.sum(gauss, dim=-1)


def polyval(p, x):
    y = torch.zeros_like(x)
    p = torch.fliplr(p.view(1, p.shape[0])).flatten()  # (1, 12)
    for i in range(len(p)):
        y = y * x + p[i]
    return y


def find_fermi(eigenvalues, highest_occupied):
    fermi = torch.max(eigenvalues[:, highest_occupied - 1])
    if fermi > 0:
        print("fermi diff", fermi)
    return fermi


if __name__ == "__main__":
    # file = "SCAN_vasprun.xml"  # file = "vasprun_NaCl.xml" "graphite.xml.bz2"
    # file = "diamond.xml.bz2"
    # file = "graphite.xml.bz2"
    # file = "MoS2.xml.bz2"
    # file = "BN.xml.bz2"
    # file = "Al.xml.bz2"
    file = "MoS2_modPBE0_vasprun.xml"
    # file = "ZnO.xml.bz2"
    # file = "Zn.xml.bz2"

    print("Device cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
        device = torch.device(dev)

    ### USER INTERFACE ###
    print_out = False
    plot = True

    cutoff_inp = 10.59544  # 20 * 0.529772  # conversion angstrom to a0 atomic unit   10.59544 # 5
    band_reducer = 8  # number_of_electrons = highest_occupied * 2  # 8
    slice_size = 1
    learning_rate = 1e-10
    num_steps = 100000
    off_set = 0  # good idea ?!?!?!?!?!?!?!?!??!?!?!?!?!?

    ### MODEL PARAMETERS ###
    spd = True  # check implementation of params_inp
    gauss = True
    model_poly = 12  # should be even
    model_gauss = 3  # should be %3 == 0

    kpts_inp, weights_inp, lattice_inp, positions_inp, species_inp, species_dict_inp, true_inp, highest_occupied = \
        eval_vasp_xml(file=file,
                      recip=False,
                      print_out=print_out,
                      slice_size=slice_size,
                      spin_up=True)

    # np.save("kpts.np", kpts_inp.detach().numpy())
    # np.save("lattice_inp", lattice_inp.detach().numpy())
    # np.save("positions_inp.np", positions_inp.detach().numpy())
    # np.save("true_inp.np", true_inp.detach().numpy())
    # np.save("species_inp.np", species_inp.detach().numpy())




    # true_up_inp = true_inp[:, :, 0]  # for one spin type in system
    true_up_inp = true_inp  # for one spin type in system


    number_of_electrons = highest_occupied * 2  # 8
    # print("electrons", number_of_electrons)


    params_tb_over_inp, params_diag_inp = eval_model(species_dict=species_dict_inp,
                                                                                 model_poly=model_poly,
                                                                                 model_gauss=model_gauss,
                                                                                 print_out=print_out,
                                                                                 gauss=gauss,
                                                                                 spd=spd)



    test_mod = DFTB(len(species_dict_inp)-1, cutoff_inp,
                    lattice=lattice_inp,
                    diag_params=params_diag_inp,
                    highest_occupied=highest_occupied,
                    init_param_values=params_tb_over_inp,
                    gauss=gauss,
                    spd=spd)

    print("occupied", highest_occupied)

    print("Optimizer")
    optim = torch.optim.AdamW(test_mod.dftb_params['V'].parameters(), lr=learning_rate)
    loss_f = torch.nn.MSELoss()
    for i in range(100000):
        # band_reducer = 2
        optim.zero_grad()
        results = test_mod.forward(positions=positions_inp, species=species_inp,
                                kpts=kpts_inp)  # [:9]
        # print(true_up_inp.shape, results.shape)
        # print(true_up_inp[0, :], results[0, :])
        # loss = loss_f(true_up_inp[:, :results.shape[1]], results[:, :])  # dftb_eig   true_up_inp
        band_count = (min(results.shape[1], true_up_inp.shape[1]) - band_reducer)
        # print("bands", band_count, true_up_inp[:, :].shape, results[:, :].shape)
        loss = loss_f(true_up_inp[:, :band_count], results[:, :band_count])  # dftb_eig   true_up_inp
        print(i, 'loss', loss)  # results
        loss.backward()
        optim.step()

        if (i % 100) == 0 and plot:
            print("plot shape", results[:9, :].shape)
            plt.plot(results[:9, :].detach().numpy(), label="Fit. Band", color="blue", lw=1)
            plt.plot(true_up_inp[:9, :].detach().numpy(), label="DFT Bands", color="red", lw=0.7)
            plt.plot(results[:9, -band_reducer:].detach().numpy(), label="Not Fit Bands", color="green", lw=0.7)
            plt.title(str("Bandstructure of " + file + ", run: " + str(i) + ", loss:" + str(loss.detach().numpy().round(decimals=2))))
            plt.ylabel("Energy")
            plt.ylim(-10, 20)
            # plt.legend()
            plt.savefig(str("Trained Bandstructure of " + file + ".png"))
            plt.show()
            plt.close()


        ### get path and band structure ###
        #
        # vs = pymatgen.io.vasp.Vasprun(file)
        # st = vs.structures[0]
        # k_path = pymatgen.symmetry.kpath.KPathSeek(st)
        # print("K path", k_path.get_kpoints(coords_are_cartesian=False)[0])
        # vs.get_band_structure()
        # # sumo.cli.bandplot.bandplot()
        # bandplot.bandplot(ymin=-20., ymax=20., boltz={"ifinter": "T", "lpfac": "10", "energy_range": "50", "curvature": "", "load": "", 'ismetaltolerance': '0.1'}, nelec=0)  # nelec=number_of_electrons


