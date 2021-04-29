""" Main Programm containing all methods and functions to compute DFTB using JAX
"""
import sys
import jax
from jax.config import config
config.update("jax_enable_x64", True)
from typing import Dict
from jax_md.util import Array
from jax_md.util import f32
from jax_md.util import f64
from jax_md.util import safe_mask
# from jax_md.smap import _kwargs_to_parameters
from jax_md.space import distance, square_distance
from jax import grad, jit, vmap, numpy as jnp
from jax.experimental import optimizers
import pandas as pd

from typing import Callable, Union, Tuple, Any
# from params_jax import get_hop_int
from params_jax_spd import get_hop_int
# from params_jax_sp import get_hop_int
import numpy as np
import scipy as scipy_nonjax
import jax.scipy as scipy
import matplotlib.pyplot as plt
import tarfile
import time
import jax.profiler
import itertools
from skfio.problem import Problem
from skfio.models import makeModel
# import seekpath
import json
# import ase
# from ase import io
import bz2
import pymatgen
from pymatgen.io.vasp import outputs
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from six.moves import cPickle as pickle
from eigh_impl import symmetrize, eigh_generalized, standardize_angle

# jax.config.update('jax_platform_name', 'cpu')


# License?

def is_hermitian(mx, TOL=1e-9):
    """
    Test whether mx is a hermitian matrix.

    Parameters
    ----------
    mx : numpy array
        Matrix to test.

    TOL : float, optional
        Tolerance on absolute magitude of elements.

    Returns
    -------
    bool
        True if mx is hermitian, otherwise False.
    """
    (m, n) = mx.shape
    for i in range(m):
        if abs(mx[i, i].imag) > TOL: return False
        for j in range(i + 1, n):
            if abs(mx[i, j] - mx[j, i].conjugate()) > TOL: return False
    return True


# @jit
def _kwargs_to_parameters(species: Array = None, **kwargs) -> Dict[str, Array]:
    """Extract parameters from keyword arguments."""
    # NOTE(schsam): We could pull out the species case from the generic case.
    s_kwargs = kwargs
    for key, val in kwargs.items():
        s_kwargs[key] = val[species]
    return s_kwargs


# @jax.profiler.trace_function
def compute_shift(lattice, cutoff):
    """ calculate neccessary repetitions in each direction with reciprocal lattice
    Args:
        lattice: takes lattice matrix as 2D-Array , e.g.: jnp.diag(jnp.ones(3))
        cutoff: cutoff distance as float
    Returns:
        shifts matrix as 2D matrix of shift vectors
    """
    n_repeat = jnp.int32(jnp.ceil(jnp.linalg.norm(jnp.linalg.inv(lattice), axis=0) * cutoff))
    # n_repeat = np.int32(np.asarray([1, 1, 1]))
    print("Repeat", n_repeat)
    relative_shifts = jnp.array([[el, el2, el3] for el in range(-n_repeat[0], n_repeat[0] + 1, 1)
                                 for el2 in range(-n_repeat[1], n_repeat[1] + 1, 1)
                                 for el3 in range(-n_repeat[2], n_repeat[2] + 1, 1)])
    relative_shifts2 = jnp.where(jnp.where(relative_shifts > 0, relative_shifts-1, relative_shifts) < 0, relative_shifts + 1,
                                jnp.where(relative_shifts > 0, relative_shifts-1, relative_shifts))
    shifts = jnp.matmul(jnp.expand_dims(lattice.T, axis=0).repeat(relative_shifts2.shape[0], axis=0),
                        jnp.expand_dims(relative_shifts2, -1)).squeeze()
    relative_shifts = relative_shifts[jnp.where(np.linalg.norm(shifts, axis=1) < cutoff)]
    shifts = jnp.matmul(jnp.expand_dims(lattice.T, axis=0).repeat(relative_shifts.shape[0], axis=0),
                            jnp.expand_dims(relative_shifts, -1)).squeeze()
    return shifts


# @jax.profiler.trace_function
# @jit
def map_product(distance):
    """ vmap is used to effectively calculate the distances of the cartesian product of the particles
    Args:
        distance: distance_fn that accepts ((N1,N2,dim), (N3,dim)) arrays as input

    Returns: map prduct of distance
    """
    return vmap(vmap(vmap(distance, (0, None), 0), (1, None), 1), (None, 0), 0)


# @jax.profiler.trace_function
# @jit
def shift_fn(r_a, shifts):
    """
    Args:
        r_a: position of particle a as vector
        shifts: uses 2D shifts matrix of comupte_shifts function
    Returns:
    """
    return jnp.repeat(jnp.expand_dims(r_a, axis=0), shifts.shape[0], axis=0) + shifts


# @jax.profiler.trace_function
# @jit
def dist_vec(r_a, r_b):
    return r_a - r_b


# @jax.profiler.trace_function
# @jit
def pair_dist_vec(r_a, r_b):
    """ Calculates Distances between two sets of vectors.
    Calculates all the distances between all possible particle combinations of the particles in r_a (N1, N2, dim) and r_b (N3,dim)
    Args:
        r_a: (N1, N2, dim) array
        r_b: (N3,dim)
    Returns:
        Matrix of distance vectors (N1, N3, N2, 3)
    """
    return map_product(dist_vec)(r_a, r_b)


def cartesian_prod(x, y):
    return jnp.stack([jnp.tile(x, len(y)), jnp.repeat(y, len(x))]).T


# @jax.profiler.trace_function
# @jit
def get_dir_cos(dist_vec):
    """ Calculates directional cosines from distance vectors.
    Calculate directional cosines with respect to the standard cartesian
    axes and avoid division by zero
    Args:
        dist_vec: distance vector between particles
    Returns: dir_cos, array of directional cosines of distances between particles
    """
    norm = distance(dist_vec)
    dir_cos = dist_vec * jnp.repeat(jnp.expand_dims(jnp.where(
        jnp.linalg.norm(dist_vec, axis=-1) == 0, jnp.zeros(norm.shape), 1 / norm), axis=-1), 3, axis=-1)
    return dir_cos


# @jax.profiler.trace_function
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
        mask = jnp.where(jnp.logical_and(dr > 0.1, dr < cutoff), 1.0, 0.0)
        return mask

    return create_bondmatrix  # mask_cutoff


# @jax.profiler.trace_function
def create_get_params_fn(species, cutoff):
    """ Creates functions to calculate params and diagonal params for a certain species number that can be jitted
    Args:
        species: 1D Numpy array of species
        cutoff: float for cutoff distance in Angstrom
    Returns: functions get_params and get_params_diag
    """
    species_count = jnp.max(species) + 1

    # @jax.profiler.trace_function
    # @jit
    def get_params(dr, species_a, species_b, kwargs):
        """
        Args:
            dr: 2D matrix of distances of particles
            species_a: element of first particle
            species_b: element of second particle
            kwargs: Dict of params for SK e.g, {"V_sss":0.2, ...}
        Returns: param_vec: slayter-koster parameters for given particles and distances
        """
        # (N_particle, N_particle, N_images, 1 -> N-parameters through broadcasting)
        param_vec = jnp.expand_dims(jnp.zeros(dr.shape), axis=-1)

        for i in range(species_count):
            for j in range(species_count):
                mask_a = jnp.array(species_a == i, dtype=dr.dtype)
                mask_b = jnp.array(species_b == j, dtype=dr.dtype)
                mask = mask_a * mask_b
                param_vec += (calc_tb_params(dr, cutoff, _kwargs_to_parameters((i, j), **kwargs))
                              * jnp.expand_dims(mask, axis=-1))  # shapes are brodcasted
        return param_vec


    # @jax.profiler.trace_function
    # @jit
    def get_params_diag(dr, species_a, kwargs_diag):
        """
        Args:
            dr: 2D matrix of distances of particles
            species_a: element of particle corresponding to parameters
            kwargs_diag: Dict of 2D matrix of one-site parameters
        Returns: param_diag: one-site parameters for given particles and distances
        """
        # (N_particle, N_particle, N_images, 1 -> N-parameters through broadcasting)
        param_diag_vec = jnp.expand_dims(jnp.zeros(dr.shape), axis=-1)

        for i in range(species_count):
            mask = jnp.array(species_a == i, dtype=dr.dtype)
            mask *= mask
            param_diag_vec += (calc_diag_params(dr, cutoff, _kwargs_to_parameters((i), **kwargs_diag))
                               * jnp.expand_dims(mask, axis=-1))
            param_diag = vmap(vmap(vmap(jnp.diag, 0), 0), 0)(param_diag_vec)
        return param_diag

    # @jax.profiler.trace_function
    # @jit
    def get_params_overlap(dr, species_a, species_b, kwargs_overlap):
        """
        Args:
            dr: 2D matrix of distances of particles
            species_a: element of first particle
            species_b: element of second particle
            kwargs: Dict of params for SK e.g, {"V_sss":0.2, ...}
        Returns: param_vec: slayter-koster parameters for given particles and distances
        """
        # (N_particle, N_particle, N_images, 1 -> N-parameters through broadcasting)
        param_vec = jnp.expand_dims(jnp.zeros(dr.shape), axis=-1)

        for i in range(species_count):
            for j in range(species_count):
                mask_a = jnp.array(species_a == i, dtype=dr.dtype)
                mask_b = jnp.array(species_b == j, dtype=dr.dtype)
                mask = mask_a * mask_b
                param_vec += (calc_overlap_params(dr, cutoff, _kwargs_to_parameters((i, j), **kwargs_overlap))
                              * jnp.expand_dims(mask, axis=-1))  # shapes are brodcasted
        return param_vec

    return get_params, get_params_diag, get_params_overlap


# @jax.profiler.trace_function
# @jit
def calc_tb_params(dr, cutoff, kwargs):
    """Select parameters for species pair from kwargs dictionary and set onsite terms to zero
    Args:
        dr: 2D matrix of distances of particles
        cutoff: float for cutoff distance in Angstrom
        kwargs: Dict of 2D matrix of slyer-koster parameters
    Returns: parameters for sk calculation
    """
    param_count = 10  # 10 for d orbitals? # 4 for S orbitalas? # maybe use 17 for l, m, n? later might be len(kwargs)
    # sk_key_list = ['Vsss', 'Vsps', 'Vpps', 'Vppp']  # , 'Vsds', 'Vpds', 'Vpdp', 'Vdds', 'Vddp', 'Vddd']
    sk_key_list = ['Vsss', 'Vsps', 'Vpps', 'Vppp', 'Vsds', 'Vpds', 'Vpdp', 'Vdds', 'Vddp', 'Vddd']
    # , 'VSSs', 'VsSs', 'VSps', 'VSds']
    param = jnp.repeat(jnp.expand_dims(jnp.zeros((dr.shape)), axis=-1), param_count, axis=-1)
    counter = 0

    for key in sk_key_list:
        # param = dist_dependent_params(dr, kwargs, key)  # return value using dist_dependent_prams
        param = param.at[:, :, :, counter].set(
            jnp.where(jnp.logical_or(dr <= 0.1, dr > cutoff), 0.0, jnp.polyval(kwargs[key][-1::-1], dr*1.88973)))
        counter += 1
    return param  # interactions[species_a, species_b]


# @jax.profiler.trace_function
# @jit
def calc_diag_params(dr, cutoff, kwargs_diag):
    """returns the onsite terms for the Hamiltonian
    Args:
        dr: 2D matrix of distances of particles
        cutoff: float for cutoff distance in Angstrom
        kwargs_diag: Dict of 2D matrix of on-site parameters
    Returns: parameters for on-site
    """
    param_count = 9  # 4 ????????
    diag_key_list = ['e_s', 'e_px', 'e_py', 'e_pz', 'e_dxy', 'e_dxz', 'e_dyz', 'e_dz2', 'e_dx2-y2']  # , 'e_S']
    param_diag = jnp.repeat(jnp.expand_dims(jnp.zeros((dr.shape)), axis=-1), param_count, axis=-1)
    counter = 0

    for key in diag_key_list:
        param_diag = param_diag.at[:, :, :, counter].set(jnp.where(dr != 0.0, 0.0, kwargs_diag[key]))
        counter += 1
    return param_diag


# @jax.profiler.trace_function
# @jit
def calc_overlap_params(dr, cutoff, kwargs_overlap):  # useless
    """Select parameters for species pair from kwargs dictionary and set onsite terms to zero
    Args:
        dr: 2D matrix of distances of particles
        cutoff: float for cutoff distance in Angstrom
        kwargs: Dict of 2D matrix of slyer-koster parameters
    Returns: parameters for sk calculation
    """
    param_count = 10  # 14 for d orbitals?  # maybe use 17 for l, m, n? later might be len(kwargs)
    # sk_key_list = ['Ssss', 'Ssps', 'Spps', 'Sppp']  # , 'Ssds', 'Spds', 'Spdp', 'Sdds', 'Sddp', 'Sddd']
    sk_key_list = ['Ssss', 'Ssps', 'Spps', 'Sppp', 'Ssds', 'Spds', 'Spdp', 'Sdds', 'Sddp', 'Sddd']

    # , 'VSSs', 'VsSs', 'VSps', 'VSds']
    param = jnp.repeat(jnp.expand_dims(jnp.zeros((dr.shape)), axis=-1), param_count, axis=-1)
    counter = 0

    for key in sk_key_list:
        # param = dist_dependent_params(dr, kwargs, key)  # return value using dist_dependent_prams
        param = param.at[:, :, :, counter].set(jnp.where(jnp.logical_or(dr <= 0.1, dr > cutoff), 0.0,
                                                         jnp.polyval(kwargs_overlap[key][-1::-1], dr*1.88973)))
        counter += 1
    return param


# @jax.profiler.trace_function
# @jit
def dist_dependent_params(dr, kwargs, key):
    param = jnp.polyval(kwargs[key], dr)
    return param


# @jax.profiler.trace_function
# @jit
def get_rec_lattice(lattice):
    return jnp.linalg.inv(lattice)


# @jax.profiler.trace_function
# @jit
def calc_phase_matrix(kpt, shift, lattice):
    """ Calculates the phases arising from a set of k-points and shifts
    Args:
        kpt: Coordinates of k-points as matrix (N_k,dim)
        shift: Matrix of shifts returned from compute_shifts function (N_shifts, dim)
        lattice: Lattice vectors as 2D matrix e.g.: jnp.array([[1.0, 0.0, 0], [0.5, jnp.sqrt(3.0)/2.0, 0], [0, 0, 10]])
    Returns:
        g_mat:
    """
    rec_lat = get_rec_lattice(lattice)
    # kpt_cart = jnp.dot(kpt, rec_lat)  # maybe transpose?
    kpt_cart = jnp.dot(rec_lat, kpt)

    # print("kpts_cart", kpt_cart.shape, kpt_cart)
    # print("shift", shift.shape, shift)
    # print("rec_lat", rec_lat.shape, rec_lat)
    g_mat = jnp.exp(2. * jnp.pi * 1j * jnp.dot(kpt_cart, shift))
    return g_mat


# @jax.profiler.trace_function
def create_hamiltonian_wo_k_fn(lattice, cutoff, get_params, get_params_diag, get_params_overlap):
    """ create function to calculate hamiltonian without phases
    Args:
        lattice: crystal lattice matrix as 2D-Array , e.g.: jnp.diag(jnp.ones(3))
        cutoff: Float for cutoff distance
        get_params: function to get slater-koster parameters for hop int
        get_params_diag: function to get on-site parametrs
    Returns: function to calculate hamiltonian_wo_k
    """

    # @jax.profiler.trace_function
    # @jit
    def create_hamiltonian_wo_k(positions, species, shifts, kwargs, kwargs_diag, kwargs_overlap):
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
        n_orbitals = 9  # 4 for sp, 9 for spd
        create_bondmatrix_mask = bondmatrix_masking(cutoff)
        shifted_positions = vmap(shift_fn, (0, None))(positions, shifts)
        shifted_pair_distance_vectors = (vmap(vmap(vmap(dist_vec, (0, None), 0), (1, None), 1), (None, 0), 0)
                                         (shifted_positions, positions))

        # expand species shape to be the same as the shifted coordinates
        shifted_species = jnp.repeat(jnp.expand_dims(species, axis=0), shifts.shape[0], axis=0).T
        # flatten first dimension for cartesian product
        shifted_species = shifted_species.reshape((shifted_species.shape[0] * shifted_species.shape[1],))
        shifted_species = cartesian_prod(shifted_species, species).T
        # separate into two vectors for particle pairs a,b and reshape to (particle number, particle_number, N_images)
        species_a = shifted_species[0].reshape(shifted_pair_distance_vectors.shape[0:-1])
        species_b = shifted_species[1].reshape(shifted_pair_distance_vectors.shape[0:-1])
        dir_cos = get_dir_cos(shifted_pair_distance_vectors)  # (particle number, particle number, N_images, 3)

        pair_distances = distance(shifted_pair_distance_vectors)  # (particle number, particle number, N_images, dim)
        bondmatrix = create_bondmatrix_mask(pair_distances)

        # off-site
        param_vec = get_params(pair_distances, species_a, species_b, kwargs)
        param_vec *= jnp.expand_dims(bondmatrix, axis=-1)
        time_start = time.time()
        hamiltonian = vmap(vmap(vmap(get_hop_int, 0), 0), 0)(jnp.concatenate([param_vec, dir_cos], axis=-1))
        time_end = time.time()
        print("Time get hop int", time_end-time_start)
        # print("hamiltonian", hamiltonian.shape)
        
        # onsite
        param_diag = get_params_diag(pair_distances, species_a, kwargs_diag)
        hamiltonian += param_diag

        # overlap matrix
        overlap_vec = get_params_overlap(pair_distances, species_a, species_b, kwargs_overlap)
        overlap_vec *= jnp.expand_dims(bondmatrix, axis=-1)
        overlap_matrix = vmap(vmap(vmap(get_hop_int, 0), 0), 0)(jnp.concatenate([overlap_vec, dir_cos], axis=-1))

        # reshape hamiltonian \ overlap to (particle number*N_orbitals, particle number*N_orbitals, N_images)
        hamiltonian = jnp.reshape(jnp.transpose(hamiltonian, (0, 3, 1, 4, 2)),
                                  (species.shape[0] * n_orbitals, species.shape[0] * n_orbitals, shifts.shape[0]))
        overlap_matrix = jnp.reshape(jnp.transpose(overlap_matrix, (0, 3, 1, 4, 2)),
                                  (species.shape[0] * n_orbitals, species.shape[0] * n_orbitals, shifts.shape[0]))
        # print("hamiltonian", hamiltonian.shape)

        # onsite overlap
        # overlap_diag = jnp.expand_dims(jnp.diag(jnp.ones(species.shape[0] * n_orbitals)), -1)
        # print("overlap", overlap_diag.shape, overlap_diag[:, :, 0])
        # overlap_matrix += overlap_diag
        # print("overlap", overlap_matrix.shape, overlap_matrix)
        # np.save("overlap_wo_k_jax.npy", overlap_matrix)

        return hamiltonian, overlap_matrix

    return create_hamiltonian_wo_k


# @jax.profiler.trace_function
# @jit
def get_ham(ham_wo_k, kpt, shifts, lattice):
    """ calculates hamiltonian for different k-points
    Args:
        ham_wo_k: hamiltonian matrix [N_p*N_o, N_p*N_o, N_images]
        kpt: Array of coordinates of the k-points, e.g. for gamma: jnp.array([[0, 0, 0]])
        shifts: uses 2D shifts matrix of compute_shifts function
        lattice: Lattice vector as 2D matrix e.g.: jnp.array([[1.0, 0.0, 0], [0.5, jnp.sqrt(3.0)/2.0, 0], [0, 0, 10]])
    Returns:
        Hamiltonian for all k-points (N_k, N_p*N_o, N_p*N_o)
    """
    phase_matrix = vmap(vmap(calc_phase_matrix, (None, 0, None)), (0, None, None))(kpt, shifts,
                                                                                   lattice)  # (N_k,N_images)
    # expand both matrices for automatic broadcasting # (N_k, 1,1, N_images)
    g_mat = jnp.expand_dims(jnp.expand_dims(phase_matrix, axis=1), axis=1)
    # print("k_kpts", kpt.shape, kpt[8, :])
    # print("g_mat", g_mat.shape, g_mat[8, :, :, :])
    # print("pre save")
    # np.save("g_mat_diamond.npy", np.asarray(g_mat))
    # print("g_mat saved")
    ham_wo_k = jnp.expand_dims(ham_wo_k, axis=0)  # (1,particle number*N_orbitals, particle number*N_orbitals, N_images)
    hamiltonian = ham_wo_k * g_mat  # (N_k, particle number*N_orbitals, particle number*N_orbitals, N_images)
    hamiltonian = jnp.sum(hamiltonian, axis=-1)  # (N_k, particle number*N_orbitals, particle number*N_orbitals)
    hamiltonian = jnp.where(jnp.abs(hamiltonian) < 1e-10, 0, hamiltonian)
    # hamiltonian += vmap(set_diagonal_to_inf, 0)(hamiltonian)  # in calculation function
    return hamiltonian


# @jax.profiler.trace_function
# @jit
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
    diag = jnp.sum(jnp.abs(hamiltonian), axis=0)
    diag = jnp.where(diag == 0, value, 0)
    return jnp.diag(diag)


# @jax.profiler.trace_function
def solve_eval(ham):
    """
    Args:
    Returns:
    """
    return sol_ham(ham, eig_vectors=False, generalized=False)


# @jax.profiler.trace_function
def solve_eval_and_evec(ham):
    """
    Args:
    Returns:
    """
    return sol_ham(ham, eig_vectors=True, generalized=False)


# @jax.profiler.trace_function
def sol_ham(ham, overlap=None, eig_vectors=False, generalized=True):
    """
    Args:
        ham: uses hamiltonian matrix from get_ham and calculates the eigenvalues/ vectors
    Returns:
        calculates the eigenvalues/ vectors and returns 1D or 1D/ 2D array
    """
    # if jnp.max(ham - ham.T.conj()) > 1.0E-9:
    #   raise Exception("\n\nHamiltonian matrix is not hermitian?!")
    if generalized == False:
        if eig_vectors == False:
            eval = scipy.linalg.eigh(ham, eigvals_only=True)
            # eval = self.clean_eig(eval)
            return jnp.array(eval)
        else:
            eval, eig = scipy.linalg.eigh(ham, eigvals_only=False)
            eig = eig.T
            # (eval, eig) = self.clean_eig(eval, eig)
            return eval, eig
    else:
        if eig_vectors == False:
            # ham = symmetrize(ham)
            # overlap = symmetrize(overlap)
            vals, vecs = eigh_generalized(ham, overlap)  # , eigvals_only=True)
            # eval = self.clean_eig(eval)
            return jnp.array(vals)
        else:
            vals, vecs = eigh_generalized(ham, overlap, eigvals_only=False)
            vecs = vecs.T
            # (eval, eig) = self.clean_eig(eval, eig)
            return vals, vecs


# @jax.profiler.trace_function
def create_calculation(lattice, species_input, cutoff):
    """ creates jittable function to create the hamiltonian and calculate its eigenvalues
    Args:
        lattice: takes lattice matrix as 2D-Array , e.g.: jnp.diag(jnp.ones(3))
        species_input: Array of species, e.g. jnp.array([0, 0]) or jnp.array([0, 1])
        cutoff: cutoff distance as float
    Returns: calculation function to create the hamiltonian and calculate its eigenvalues
    """
    get_params, get_params_diag, get_params_overlap = create_get_params_fn(species_input, cutoff)
    create_hamiltonian_wo_k = create_hamiltonian_wo_k_fn(lattice, cutoff, get_params, get_params_diag, get_params_overlap)
    # @jit
    # @jax.profiler.trace_function
    def calculation(lattice, positions, species, shifts, kpts, kwargs, kwargs_diag, kwargs_overlap):
        """ creates function to return parameters and hamiltonian_wo_k
        Args:
            lattice: takes lattice matrix as 2D-Array , e.g.: jnp.diag(jnp.ones(3))
            positions: Array of position vectors of atoms
            species: Array of species, e.g. jnp.array([0, 0]) or jnp.array([0, 1])
            shifts: uses 2D shifts matrix of comupte_shifts function
            kpts: Array of coordinates of the k-points, e.g. for gamma: jnp.array([[0, 0, 0]])
        Returns: 2D- vector of eigenvalues from hamiltonian for each k point
        """
        ham_wo_k, overlap_wo_k = create_hamiltonian_wo_k(positions, species, shifts, kwargs, kwargs_diag, kwargs_overlap)
        hamiltonian = get_ham(ham_wo_k, kpts, shifts, lattice)
        hamiltonian += vmap(set_diagonal_to_inf, 0)(hamiltonian)
        # print("hamiltonian", hamiltonian.shape, jnp.round(jnp.abs(hamiltonian[0, :, :]), decimals=2))  # , jnp.iscomplex(hamiltonian))  # , "\n", hamiltonian[0, :, :])
        overlap_matrix = get_ham(overlap_wo_k, kpts, shifts, lattice)
        # print("overlap", jnp.expand_dims(jnp.diag(jnp.ones(overlap_matrix.shape[1])), 0).shape)
        overlap_matrix += jnp.expand_dims(jnp.diag(jnp.ones(overlap_matrix.shape[1])), 0)

        solution_jaxscipy = scipy.linalg.eigh(hamiltonian, eigvals_only=True)
        # print("Solutions jax scipy", solution_jaxscipy.shape, solution_jaxscipy)

        # to calculate generalized eigenvalue problem
        hamiltonian = jnp.where(jnp.abs(hamiltonian) < 10e-10, 0, hamiltonian)
        overlap_matrix = jnp.where(jnp.abs(overlap_matrix) < 10e-10, 0, overlap_matrix)
        overlap_inverse = vmap(jnp.linalg.inv, 0)(overlap_matrix)
        new_ham = vmap(jnp.dot, 0, 0)(overlap_inverse, hamiltonian)

        # solution_generalized, vectors = eigh_generalized(hamiltonian, overlap_matrix)
        # print("Solutions generalized", solution_generalized.shape, solution_generalized[0, :])

        # solution = sol_ham(new_ham[1, :, :], eig_vectors=False, generalized=False)
        solution_jaxscipy_gen = scipy.linalg.eigh(new_ham, eigvals_only=True)
        # print("Solutions gen jax scipy", solution_jaxscipy_gen.shape, solution_jaxscipy_gen[0, :])

        # solution_jaxscipy_gen_np = scipy_nonjax.linalg.eigh(new_ham[0, :, :], eigvals_only=True)
        # print("Solutions gen numpy scipy", solution_jaxscipy_gen_np.shape, solution_jaxscipy_gen_np)

        # solution_jaxnumpy_gen, _ = jnp.linalg.eigh(new_ham)
        # print("Solutions gen jax numpy", solution_jaxnumpy_gen.shape, solution_jaxnumpy_gen[0, :])
        solution_jaxscipy_gen -= find_fermi(solution_jaxscipy_gen, highest_occupied, plot=False)
        solution_jaxscipy_gen = solution_jaxscipy_gen * 27.211396  # conversion au (atomic unit) to eV
        return solution_jaxscipy_gen  # -[:, -1::-1]  # [:9, :8]

    return calculation


# @jax.profiler.trace_function
def create_loss_fn(lattice, positions, species, shifts, kpts, kwargs, kwargs_diag, kwargs_overlap, true):
    """ calculates loss between true eigenvalues and calculated ones
    Args:
    Returns: sum of squared difference of eigenvalues
    """
    def loss_fn_kwargs(kwargs):
        number_of_electrons = 4  # number_of_electrons:2*number_of_electrons ???
        return jnp.mean((true[:, :number_of_electrons] - calculation(lattice, positions, species, shifts, kpts,
                                           kwargs, kwargs_diag, kwargs_overlap)[:, :number_of_electrons]) ** 2)  # [:, true.shape[1]]

    def loss_fn_kwargs_overlap(kwargs_overlap):
        return jnp.mean((true - calculation(lattice, positions, species, shifts, kpts,
                                           kwargs, kwargs_diag, kwargs_overlap)[:, :number_of_electrons]) ** 2)  # [:, true.shape[1]]

    def loss_fn(kwargs, kwargs_overlap):
        return jnp.mean((true - calculation(lattice, positions, species, shifts, kpts,
                                           kwargs, kwargs_diag, kwargs_overlap)[:, :number_of_electrons]) ** 2)  # [:, true.shape[1]]
    return loss_fn_kwargs, loss_fn_kwargs_overlap, loss_fn


def gauss_func(x, alpha, beta):
    """
    func to crate a gauss function
    :param x:  input grid array
    :param alpha: decay factor in exponent
    :param beta: pre-factor of exponetial func
    :return: gauss type func
    """

    return beta * jnp.exp(- alpha * x ** 2)


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def find_fermi(eigenvalues, highest_occupied, plot=False):
    fermi = jnp.max(eigenvalues[:, highest_occupied - 1])
    if plot:
        plt.ylim(-100, 100)
        plt.plot(eigenvalues[:, highest_occupied - 1], label="Band")
        plt.plot(np.arange(0, eigenvalues.shape[0]), np.repeat(fermi, eigenvalues.shape[0]), label="Band")
        plt.show()
    return fermi


def eval_vasp_xml(file="vasprun.xml", recip=False, norm_fermi=True, print_out=False):
    dft = pymatgen.io.vasp.outputs.Vasprun(file, parse_projected_eigen=False)
    orbital_energy = pd.read_csv("element_orbital_energy.csv").set_index("element")

    lattice = jnp.asarray(dft.get_trajectory().as_dict()['lattice']).squeeze()
    lattice_normed = lattice / jnp.linalg.norm(lattice, axis=1, keepdims=True)
    lattice_recip = jnp.asarray(Lattice(lattice).reciprocal_lattice.matrix)  # wrong!

    positions_base = dft.get_trajectory().as_dict()['base_positions']
    positions = jnp.dot(positions_base, lattice)

    k_points = jnp.asarray(dft.actual_kpoints)

    weights = jnp.asarray(dft.actual_kpoints_weights)  # how to use ?

    species_dict = {}
    species_arr = np.asarray(dft.atomic_symbols)
    count = 0
    print(species_arr)
    for key in dict.fromkeys(set(dft.atomic_symbols), {}):
        species_dict["species_" + Element(key).long_name] = {"symbol": key,
                                                             "number": count,
                                                             "Es": orbital_energy.loc["C", "E_s"],
                                                             "Ep": orbital_energy.loc["C", "E_p"],
                                                             "Ed": orbital_energy.loc["C", "E_d"],
                                                             }
        species_arr[species_arr == key] = count  # cycles through elements but returns correct one anyway
        count += 1
    species_arr = jnp.asarray(species_arr.astype(int))

    for key in dft.eigenvalues.keys():
        key_last = key
    true_inp = np.zeros(
        (dft.eigenvalues[key_last][:, :, 0].shape[0], dft.eigenvalues[key_last][:, :, 0].shape[1], len(dft.eigenvalues.keys())))
    count = 0
    if len(dft.eigenvalues.keys()) != 1:
        print("only one spin direction supported but", len(dft.eigenvalues.keys()), "where given")
    for key in dft.eigenvalues.keys():  # OrderedDictionary might be nice
        true_inp[:, :, count] = dft.eigenvalues[key][:, :, 0]  # what is [:, :, 0] ???????????????????
        occupied = np.max(jnp.nonzero(dft.eigenvalues[key][:, :, 1])[1]) + 1
        fermi = find_fermi(true_inp, occupied)
        count += 1
    if norm_fermi:
        true_inp -= fermi
        print("E fermi calculated normed", find_fermi(true_inp, occupied, plot=False))
    if print_out:
        print("Lattice", type(lattice), lattice.shape, "\n", lattice)
        print("Lattice Normed", type(lattice_normed), lattice_normed.shape, lattice_normed)
        print("Lattice recip", type(lattice_recip), lattice_recip.shape, "\n", lattice_recip)
        print("Positions", type(positions_base), positions_base.shape, positions_base)
        print("Positions dot", type(positions), positions.shape, "\n", positions)
        print("kpts", k_points.shape, k_points)
        print("weights", weights.shape, weights)
        print("True shape", true_inp.shape, true_inp)
        print("species", species_arr.shape, species_arr, "\n", species_dict)
        # print("true", dft.eigenvalues[:].shape, "\n", dft.eigenvalues[dft.eigenvalues.keys()[0]][0, :, 0], "\n",
        #       dft.eigenvalues[dft.eigenvalues.keys()[0]][0, :, 1])
        print("E fermi vasp", dft.efermi)
        print("Highest occupied", occupied)
        print("E fermi calculated", fermi)
    if recip:
        return k_points, weights, lattice_recip, positions, species_arr, species_dict, true_inp, occupied
    else:
        return k_points, weights, lattice, positions, species_arr, species_dict, true_inp, occupied



if __name__ == "__main__":
    # server = jax.profiler.start_server(9999)

    # jnp.set_printoptions(threshold=sys.maxsize)

    # file = "SCAN_vasprun.xml"  # file = "vasprun_NaCl.xml" "graphite.xml.bz2"
    file = "diamond.xml.bz2"
    # file = "graphite.xml.bz2"
    # file = "MoS2.xml.bz2"
    # file = "BN.xml.bz2"
    # file = "Al.xml.bz2"

    kpts_inp, weights_inp, lattice_inp, positions_inp, species_inp, species_dict_inp, true_inp, highest_occupied = \
        eval_vasp_xml(file, recip=False, print_out=True)

    true_up_inp = true_inp[:, :, 0]  # for one spin type in system


    cutoff_inp = 2  #  20 * 0.529772  # conversion angstrom to a0 atomic unit

    number_of_electrons = 6  # 8
    # highest_occupied = 4

    slice_size = 1

    learning_rate = 1e-16
    num_steps = 10000

    if slice_size > 1:
        print("Kpts reduced from", kpts_inp.shape, "to:", kpts_inp[0:kpts_inp.shape[0]:slice_size, :].shape)
        kpts_inp = kpts_inp[0:kpts_inp.shape[0]:slice_size, :]
        true_inp = true_inp[0:true_inp.shape[0]:slice_size, :, :]
        true_up_inp = true_up_inp[0:true_up_inp.shape[0]:slice_size, :]

    model = "P_12"  # "P_15 * G_3"  # poly should be even
    species_dict_inp["models"] = {
            "global": True,
            "model": model
            }
    pr = Problem.fromDictionary(**species_dict_inp)

    modelData = pr.makeModelData(model, ir0=19)
    print("Modeldata", modelData.keys())

    kwargs_inp = {}
    kwargs_overlap_inp = {}
    for key in modelData:
        if key[0] == "V":
            kwargs_inp[key] = jnp.asarray(modelData[key])

        elif key[0] == "S":
            kwargs_overlap_inp[key] = jnp.asarray(modelData[key])

    print("kwargs", kwargs_inp)
    print("kwargs overlap", kwargs_overlap_inp)

    # sk_key_list = ['Vsss', 'Vsps', 'Vpps', 'Vppp', 'Vsds', 'Vpds', 'Vpdp', 'Vdds', 'Vddp', 'Vddd', 'VSSs', 'VsSs', 'VSps', 'VSds']
    # overlap_key_list = ['Ssss', 'Ssps', 'Spps', 'Sppp', 'Ssds', 'Spds', 'Spdp', 'Sdds', 'Sddp', 'Sddd', 'SSSs', 'SsSs', 'SSps', 'SSds']
    # for keyV, keyS in zip(sk_key_list, overlap_key_list):

    species_count = len(species_dict_inp) - 1
    orbital_energy = np.zeros((species_count, 3))

    for key in species_dict_inp:
        if key != "models":
            orbital_energy[species_dict_inp[key]["number"], 0] = species_dict_inp[key]["Es"]
            orbital_energy[species_dict_inp[key]["number"], 1] = species_dict_inp[key]["Ep"]
            orbital_energy[species_dict_inp[key]["number"], 2] = species_dict_inp[key]["Ed"]


    kwargs_diag_inp = {'e_s': orbital_energy[:, 0].reshape((species_count, 1)),
                       'e_px': orbital_energy[:, 1].reshape((species_count, 1)),
                       'e_py': orbital_energy[:, 1].reshape((species_count, 1)),
                       'e_pz': orbital_energy[:, 1].reshape((species_count, 1)),
                       'e_dxy': orbital_energy[:, 2].reshape((species_count, 1)),
                       'e_dxz': orbital_energy[:, 2].reshape((species_count, 1)),
                       'e_dyz': orbital_energy[:, 2].reshape((species_count, 1)),
                       'e_dz2': orbital_energy[:, 2].reshape((species_count, 1)),
                       'e_dx2-y2': orbital_energy[:, 2].reshape((species_count, 1)),
                       'e_S': jnp.zeros((species_count)).reshape((species_count, 1))
                        }


    print("kwargs_diag", kwargs_diag_inp['e_s'].shape, kwargs_diag_inp['e_s'], "len(orbital_energy[:, 0])", len(orbital_energy[:, 0]))

    # kpts_inp = jnp.asarray([[0.5,     0.,      0.    ]])
    # species_inp = jnp.asarray([0, 0])
    # # lattice_inp = jnp.asarray([[1.0, 0.0, 0], [0.5, jnp.sqrt(3.0)/2.0, 0], [0, 0, 10]])
    # lattice_inp = jnp.asarray([[2.0, 0.0, 0], [0.0, 2.0, 0], [0, 0, 2.0]])
    # positions_inp = jnp.asarray([[1., 0, 0], [0.0, 1., 0]])

    error = np.zeros(num_steps)
    eigenvalues = np.zeros((num_steps, np.shape(true_up_inp)[0], number_of_electrons))
    parameters_tb = kwargs_inp
    parameters_overlap = kwargs_overlap_inp

    shifts_inp = compute_shift(lattice_inp, cutoff_inp)
    print("Shifts shape", shifts_inp.shape)

    calculation = create_calculation(lattice_inp, species_inp, cutoff_inp)

    # g = jax.jacfwd(calculation, (1, 6, 7))

    loss_fn_kwargs, loss_fn_kwargs_overlap, loss_fn = create_loss_fn(lattice_inp, positions_inp, species_inp, shifts_inp, kpts_inp,
                                 kwargs_inp, kwargs_diag_inp, kwargs_overlap_inp, true_up_inp[:, :number_of_electrons])

    # Optimizer

    # opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_state_1 = opt_init(kwargs_inp)
    opt_state_2 = opt_init(kwargs_overlap_inp)
    # opt_state_all = opt_init(kwargs_inp, kwargs_overlap_inp)

    def step(step, opt_state, save=False, plot=False):
        # value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state_1))
        value = calculation(lattice_inp, positions_inp, species_inp, shifts_inp, kpts_inp,
                                     get_params(opt_state), kwargs_diag_inp, kwargs_overlap_inp)
        grads = jax.jacfwd(loss_fn_kwargs)(get_params(opt_state))
        # print("get_params(opt_state)", get_params(opt_state))
        # print("grads", grads)
        print(step, "error: ", jnp.mean((true_up_inp[:, :number_of_electrons] - value[:, :number_of_electrons]) ** 2))
        if save:
            save_dict(get_params(opt_state), str(file + "_params.pickle"))
            error[step] = jnp.mean((true_up_inp[:, :number_of_electrons] - value[:, :number_of_electrons]) ** 2)
            np.save(str(file + "_error" + ".npy"), error[:step])
            eigenvalues[step, :, :] = value[:, :number_of_electrons]
            np.save(str(file + "_eigenvalues" + ".npy"), eigenvalues[:step, :, :])
        if step % 5 == 0 and step != 0 and plot:
            plt.plot(error[:step], label="error", color="blue")
            plt.plot(np.gradient(error[:step], axis=0), label="derivative Error", color="red")
            plt.savefig(str(file + "_error" + ".png"))
            plt.title("Error")
            plt.legend()
            plt.ylim(-2, error.max()*1.1)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.show()
            plt.close()
        # for key, val in get_params(opt_state).items():  # tb params
        #     print("\t", key, get_params(opt_state)[key], grads[key])
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    def step2(step, opt_state):
        value = calculation(lattice_inp, positions_inp, species_inp, shifts_inp, kpts_inp,
                                     kwargs_inp, kwargs_diag_inp, get_params(opt_state))
        grads = jax.jacfwd(loss_fn_kwargs_overlap)(get_params(opt_state))
        print(step, "error: ", jnp.mean((true_up_inp[:, :number_of_electrons] - value[:, :number_of_electrons]) ** 2))
        opt_state = opt_update(step, grads, opt_state)

        return value, opt_state

    # def step_all(step, opt_state_1, opt_state_2):
    #     value = calculation(lattice_inp, positions_inp, species_inp, shifts_inp, kpts_inp,
    #                                  kwargs_inp, kwargs_diag_inp, get_params(opt_state))
    #     grads = jax.jacfwd(loss_fn_kwargs_overlap)(get_params(opt_state_1), get_params(opt_state_2))
    #     print(step, "error: ", jnp.mean((true_inp[:, :8] - value[:, :8]) ** 2))
    #     opt_state = opt_update(step, grads, opt_state)
    #     return value, opt_state

    for i in range(num_steps):
        time_total_start = time.time()
        value, opt_state_1 = step(i, opt_state_1, save=True, plot=True)
        time_total_end = time.time()
        print("Time Total Epoch", time_total_end - time_total_start)
        # value, opt_state_2 = step2(i, opt_state_2)
        # value, opt_state_1, opt_state_2 = step_all(i, opt_state_1, opt_state_2)
        if i % 5 == 0:
            plt.plot(value[:9, :number_of_electrons], label="Fit. Band", color="blue", lw=1)
            plt.plot(true_up_inp[:9, :number_of_electrons], label="DFT Bands", color="red", lw=0.7)
            plt.title(str("Bandstructure of " + file + " run " + str(i)))
            plt.ylabel("Energy")
            # plt.legend()
            plt.savefig(str("Trained Bandstructure of " + file + ".png"))
            plt.show()
            plt.close()

            # fig, ax = plt.subplots()
            # x = np.linspace(0, 9, 9)
            # ax.set_title('Click on legend line to toggle line on/off')
            # line1, = ax.plot(true_inp[:9, :number_of_electrons], label='True Band')  # , lw=2
            # line1, = ax.plot(true_inp[0][:9], true_inp[1][:number_of_electrons], label='True Band')  # , lw=2
            # line1, = ax.plot(value[:9, :number_of_electrons], label="Band")
            # line2, = ax.plot(value[:9, :number_of_electrons], label='Fit. Band')
            # leg = ax.legend(fancybox=True, shadow=True)
            # plt.show()

