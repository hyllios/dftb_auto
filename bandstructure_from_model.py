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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# matplotlib.use( 'tkagg' )

# from pytorch_relevant import find_fermi

import pytorch_lightning as pl
from lightning_module import LitModel

# from __future__ import unicode_literals
from pkg_resources import Requirement, resource_filename

import os
import sys
import glob
import logging
import argparse
import warnings
import json
import numpy as np
from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import \
    get_reconstructed_band_structure

# import matplotlib as mpl
# mpl.use('Agg')

from sumo.io.questaal import QuestaalInit, QuestaalSite, labels_from_syml
from sumo.io.questaal import band_structure as questaal_band_structure
from sumo.plotting.bs_plotter import SBSPlotter
from sumo.plotting.dos_plotter import SDOSPlotter
from sumo.electronic_structure.dos import load_dos
from sumo.electronic_structure.bandstructure import string_to_spin
from sumo.cli.dosplot import _atoms, _el_orb

from pymatgen.electronic_structure.boltztrap2 import VasprunBSLoader, BztInterpolator
from pymatgen.electronic_structure.boltztrap2 import BztTransportProperties

from pymatgen.electronic_structure.boltztrap import BoltztrapRunner

import bandplot
import bandplot_copy
import bandplot_old


def find_fermi(eigenvalues, highest_occupied):
    fermi = np.max(eigenvalues[:, highest_occupied - 1])
    if fermi > 0:
        print("fermi diff", fermi)
    return fermi


def bandplot_func(filenames=None, code='vasp', prefix=None, directory=None,
             vbm_cbm_marker=False, projection_selection=None, mode='rgb', pred=None,
             interpolate_factor=4, circle_size=150, dos_file=None,
             cart_coords=False, scissor=None,
             ylabel='Energy (eV)', dos_label=None,
             elements=None, lm_orbitals=None, atoms=None, spin=None,
             total_only=False, plot_total=True, legend_cutoff=3, gaussian=None,
             height=None, width=None, ymin=-6., ymax=6., colours=None,
             yscale=1, style=None, no_base_style=False,
             image_format='pdf', dpi=400, plt=None, fonts=None,
             boltz={"ifinter":"T","lpfac":"10","energy_range":"50","curvature":"","load":"T",'ismetaltolerance':'0.01'},
             nelec=0):
    if not filenames:
        filenames = find_vasprun_files()
    elif isinstance(filenames, str):
        filenames = [filenames]

    # only load the orbital projects if we definitely need them
    parse_projected = True if projection_selection else False

    # now load all the band structure data and combine using the
    # get_reconstructed_band_structure function from pymatgen
    bandstructures = []
    if code == 'vasp':
        for vr_file in filenames:
            vr = BSVasprun(vr_file, parse_projected_eigen=parse_projected)
            print("BSVasprun", type(vr), vr)
            print("vr.eigenvalues.keys()", type(vr.eigenvalues.keys()), vr.eigenvalues.keys())
            if pred.any():
                # Fill in Model prediction
                model = BSVasprun(vr_file, parse_projected_eigen=parse_projected)
                print("pred", type(pred), pred.shape)
                pred = np.expand_dims(pred, axis=-1)
                for key in model.eigenvalues.keys():
                    key_last = key
                    print("model.eigenvalues[key][:, :, :].shape[0]", key,
                          type(model.eigenvalues[key][:, :, :]),
                          model.eigenvalues[key][:, :, :].shape,
                          pred[:, :, :].shape)
                    bands = min(model.eigenvalues[key][:, :, :].shape[1], pred.shape[1])
                    print("bands", bands, "attention! max: ", max(model.eigenvalues[key][:, :, :].shape[1], pred.shape[1]))
                    print("equel False?", np.sum(model.eigenvalues[key][:, :bands, :] - pred[:, :bands, :]))
                    model.eigenvalues[key][:, :bands, :] = pred[:, :bands, :]
                    print("equel True?", np.sum(model.eigenvalues[key][:, :bands, :] - pred[:, :bands, :]))
                    print("equel model vr?", np.sum(model.eigenvalues[key][:, :bands, :] - vr.eigenvalues[key][:, :bands, :]))
                    # spin = 1  # for only plotting spin up oder down 1, -1
                # model.eigenvalues[key_last][:, :bands, :] = pred[:, :bands, :]

            #boltztrap={'ifinter':False,'lpfac':10,'energy_range':50,'curvature':False}):

            if bool(boltz['ifinter']):
                b_data = VasprunBSLoader(vr)
                model_data = VasprunBSLoader(model)
                print("BSVasprunLoader", type(b_data), b_data)
                b_inter = BztInterpolator(b_data, lpfac=int(boltz['lpfac']), energy_range=float(boltz['energy_range']),
                                          curvature=bool(boltz['curvature']), save_bztInterp=True,
                                          load_bztInterp=bool(boltz['load']))
                model_inter = BztInterpolator(model_data, lpfac=int(boltz['lpfac']), energy_range=float(boltz['energy_range']),
                                          curvature=bool(boltz['curvature']), save_bztInterp=True,
                                          load_bztInterp=bool(boltz['load']))

                try:
                    kpath = json.load(open('./kpath','r'))
                    kpaths = kpath['path']
                    kpoints_lbls_dict = {}
                    for i in range(len(kpaths)):
                        for j in [0,1]:
                            if 'GAMMA'==kpaths[i][j]:
                                kpaths[i][j] = '\Gamma'
                    for k,v in kpath['kpoints_rel'].items():
                        if k=='GAMMA':
                            k='\Gamma'
                        kpoints_lbls_dict[k] = v
                except:
                    kpaths = None
                    kpoints_lbls_dict = None

                print(kpaths, kpoints_lbls_dict)
                bs = b_inter.get_band_structure(kpaths=kpaths, kpoints_lbls_dict=kpoints_lbls_dict)
                model_bs = model_inter.get_band_structure(kpaths=kpaths, kpoints_lbls_dict=kpoints_lbls_dict)

                #bs_uniform = b_inter.get_band_structure()
                gap = bs.get_band_gap()
                nvb = int(np.ceil(nelec / (int(bs.is_spin_polarized) + 1)))
                vbm = -100
                print("WHC interpolated gap: %s" %gap)
                for spin, v in bs.bands.items():
                    vbm = max(vbm, max(v[nvb-1]))
                print('WHC WARNNING vasp fermi %s interpolation vbm %s nelec %s nvb %s'%(bs.efermi,vbm,nelec,nvb))
                if vbm < bs.efermi:
                    bs.efermi = vbm
                    print("if vbm <")
                if vbm < model_bs.efermi:
                    model_bs.efermi = vbm
                    print("if vbm <")
                print(bs.bands.keys())
                band_keys = list(bs.bands.keys())
                print("Band shapes", bs.bands[band_keys[0]].shape, model_bs.bands[band_keys[0]].shape)
                print("equel bands?", np.sum((bs.bands[band_keys[0]] - bs.efermi) - model_bs.bands[band_keys[0]]))
                bs.bands[band_keys[0]] = (bs.bands[band_keys[0]] - bs.efermi)  # why??????????????????????????????????????????????????
                # bs.bands[band_keys[1]] = (bs.bands[band_keys[1]] - bs.efermi)  # why??????????????????????????????????????????????????
                print("equel bands fermi shifted?", np.sum((bs.bands[band_keys[0]] - bs.efermi) - model_bs.bands[band_keys[0]]))

                # bandstructures.append(bs)
                # bandstructures.append(model_bs)
            bs = get_reconstructed_band_structure([bs])
            model_bs = get_reconstructed_band_structure([model_bs])

            if bool(boltz['ifinter']):
                bs.nvb = nvb
                bs.ismetaltolerance = float(boltz['ismetaltolerance'])
                model_bs.nvb = nvb
                model_bs.ismetaltolerance = float(boltz['ismetaltolerance'])

            print("dft bands", bs.bands[band_keys[0]])
            print("dft ktps", len(bs.kpoints), kpts.shape)
            print("dft labels", bs.labels_dict)

            for key in bs.labels_dict.keys():
                print(bs.labels_dict[key].label,
                      bs.labels_dict[key].as_dict(),
                      bs.labels_dict[key].a,
                      bs.labels_dict[key].b,
                      bs.labels_dict[key].c,
                      bs.labels_dict[key].frac_coords
                      )
            labels = []
            for i in range(len(bs.kpoints)):
                # print(i, bs.kpoints[i])
                for key in bs.labels_dict.keys():
                    if bs.labels_dict[key].label == bs.kpoints[i].label and bs.labels_dict[key].label != bs.kpoints[i-1].label:
                        # print("Labels!!!!!", i, bs.labels_dict[key].label)
                        labels.append([i, bs.labels_dict[key].label])
            print(labels)

            print("dft efermi", bs.efermi)
            print("dft lattice_rec", bs.lattice_rec)
            print("dft structure", bs.structure)

            print("model bands", model_bs.bands[band_keys[0]])
            print("model ktps", len(model_bs.kpoints), kpts.shape)
            print("model labels", model_bs.labels_dict)
            print("model efermi", model_bs.efermi)
            print("model lattice_rec", model_bs.lattice_rec)
            print("model structure", model_bs.structure)


            return bs.bands[band_keys[0]], model_bs.bands[band_keys[0]], labels


            save_files = False if plt else True
            dos_plotter = None
            dos_opts = None
            if dos_file:
                dos, pdos = load_dos(dos_file, elements, lm_orbitals, atoms, gaussian,
                                     total_only)
                dos_plotter = SDOSPlotter(dos, pdos)
                dos_opts = {'plot_total': plot_total, 'legend_cutoff': legend_cutoff,
                            'colours': colours, 'yscale': yscale}

            model_and_dft_bs = [bs, model_bs]
            plotter = SBSPlotter(model_bs)
            print("spin", spin)
            if len(vr.eigenvalues.keys()) == 1:
                spin = None
            print("spin", spin)
            plt = plotter.get_plot(
                zero_to_efermi=True, ymin=ymin, ymax=ymax, height=height,
                width=width, vbm_cbm_marker=vbm_cbm_marker, ylabel=ylabel,
                plt=plt, dos_plotter=dos_plotter, dos_options=dos_opts,
                dos_label=dos_label, fonts=fonts, style=style,
                no_base_style=no_base_style, spin=spin)

        # don't save if pyplot object provided
        save_files = False if plt else True
        if save_files:
            basename = 'band.{}'.format(image_format)
            filename = '{}_{}'.format(prefix, basename) if prefix else basename
            if directory:
                filename = os.path.join(directory, filename)
            plt.savefig(filename, format=image_format, dpi=dpi,
                        bbox_inches='tight')

            written = [filename]
            written += save_data_files(bs, prefix=prefix,
                                       directory=directory)
            return written

        else:
            return plt


# filenames=None
# code='vasp'
# prefix=None
# directory=None
# vbm_cbm_marker=False
# projection_selection=None
# mode='rgb'
# pred=None
# interpolate_factor=4
# circle_size=150
# dos_file=None
# cart_coords=False
# scissor=None,
# ylabel='Energy (eV)'
# dos_label=None,
# elements=None
# lm_orbitals=None
# atoms=None
# spin=None
# total_only=False
# plot_total=True
# legend_cutoff=3
# gaussian=None
# height=None
# width=None
# ymin=-6.
# ymax=6.
# colours=None
# yscale=1
# style=None
# no_base_style=False
# image_format='pdf'
# dpi=400
# plt=None
# fonts=None
boltz={"ifinter":"T","lpfac":"10","energy_range":"50","curvature":"","load":"T",'ismetaltolerance':'0.01'}
nelec=0

band_reducer = 0

# Load model from checkpoint
model = LitModel.load_from_checkpoint('test_checkpoint.ckpt')
highest_occupied = model.highest_occupied.detach().numpy()
print("highest_occupied", highest_occupied)
print("model hyper parameters", model.hparams)
band_reducer = model.hparams.band_reducer
file = model.hparams.file
kpts = model.dataset.kpts
true = model.dataset.true
model_results = model.forward(kpts)
print(true.shape, model_results.shape)
print(torch.sum(true[:, :8] - model_results[:, :8]))
np_model_results = model_results.detach().numpy()


### get path and band structure ###

vs = pymatgen.io.vasp.Vasprun(file)
st = vs.structures[0]
k_path = pymatgen.symmetry.kpath.KPathSeek(st)
print("K path", k_path.get_kpoints(coords_are_cartesian=False)[0])
vs.get_band_structure()
# sumo.cli.bandplot.bandplot()

dft_band, model_band, labels = bandplot_func(filenames=file,
                        prefix=str(file + "_test"),
                        # save_files=False,
                        # pred=None,
                        pred=np_model_results,
                        # spin=-1,  # -1 for model and 1 for vasp calc
                        ymin=-20.,
                        ymax=20.,
                        # plt=ax,
                        boltz={"ifinter": "T",
                               "lpfac": "10",
                               "energy_range": "50",
                               "curvature": "",
                               "load": "",
                               'ismetaltolerance': '0.1'},
                        nelec=0,
                        )  # nelec=number_of_electrons


dft_band = np.transpose(dft_band, (1, 0))
model_band = np.transpose(model_band, (1, 0))
labels_index = np.array(labels)[:, 0].astype(np.int)
labels = np.array(labels)[:, 1]
for i in range(len(labels)):
    labels[i] = str("$" + labels[i] + "$")
print(labels_index)
print(labels)

ymin = -18
ymax = 17

custom_lines = [Line2D([0], [0], color="blue", lw=4),
                Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="m", lw=4),
                Line2D([0], [0], color="red", lw=4)]

plt.plot(model_band, label="Fit. Band", color="blue", lw=0.5)
plt.plot(dft_band, label="DFT Bands", color="red", lw=0.3)
plt.plot(model_band[:, -band_reducer:], label="Not Fit Bands", color="green", lw=0.5)
plt.plot(model_band[:, (highest_occupied-1)], label="Highest Occupied", color="m", lw=0.7)
plt.title(str("Bandstructure of " + file))
plt.ylabel("Energy")
plt.ylim(ymin, ymax)
# plt.minorticks_on()
plt.xticks(labels_index, labels)
plt.vlines(labels_index, ymin, ymax, linestyles="dashed")
plt.legend(custom_lines, ["Model Fit", "Model not Fit.", "Highest Occupied", "DFT"])
plt.savefig(str("Bandstructure_Test"), dpi=400)
plt.show()
plt.close()


# bandplot.bandplot(filenames=file,
#                         prefix=str(file + "_test"),
#                         # save_files=False,
#                         # pred=None,
#                         pred=true,
#                         # spin=-1,  # -1 for model and 1 for vasp calc
#                         ymin=-20.,
#                         ymax=20.,
#                         # plt=ax,
#                         boltz={"ifinter": "T",
#                                "lpfac": "10",
#                                "energy_range": "50",
#                                "curvature": "",
#                                "load": "",
#                                'ismetaltolerance': '0.1'},
#                         nelec=0,
#                         )  # nelec=number_of_electrons
#
# print("return plot now")
#
# plt, written = bandplot_copy.bandplot(filenames=file,
#                         prefix=file,
#                         # save_files=False,
#                         # pred=None,
#                         # pred=true,
#                         # spin=-1,  # -1 for model and 1 for vasp calc
#                         ymin=-20.,
#                         ymax=20.,
#                         # plt=ax,
#                         boltz={"ifinter": "T",
#                                "lpfac": "10",
#                                "energy_range": "50",
#                                "curvature": "",
#                                "load": "",
#                                'ismetaltolerance': '0.1'},
#                         nelec=0,
#                         )  # nelec=number_of_electrons
#
#
# # print(written)
# if not file:
#     filenames = bandplot.find_vasprun_files()
# elif isinstance(file, str):
#     filenames = [file]
#
# # now load all the band structure data and combine using the
# # get_reconstructed_band_structure function from pymatgen
# bandstructures = []
#
# for vr_file in filenames:
#     vr = BSVasprun(vr_file, parse_projected_eigen=False)
#     # Replace eigenvalues with Model prediction
#     model = vr
#     print("pred", type(np_true), np_true.shape)
#     pred = np.expand_dims(np_true, axis=-1)
#     for key in model.eigenvalues.keys():
#         key_last = key
#         print("model.eigenvalues[key][:, :, :].shape[0]", key,
#               type(model.eigenvalues[key][:, :, :]),
#               model.eigenvalues[key][:, :, :].shape[0],
#               model.eigenvalues[key][:, :, :].shape[1])
#         bands = min(model.eigenvalues[key][:, :, :].shape[1], pred.shape[1])
#         print("bands", bands, "attention! max: ", max(model.eigenvalues[key][:, :, :].shape[1], pred.shape[1]))
#         print("equel", np.sum(model.eigenvalues[key][:, :bands, :] - pred[:, :bands, :]))
#         # model.eigenvalues[key][:, :bands, :] = pred[:, :bands, :]
#         # spin = 1  # for only plotting spin up oder down 1, -1
#     # model.eigenvalues[key_last][:, :bands, :] = pred[:, :bands, :]
#
# # boltztrap={'ifinter':False,'lpfac':10,'energy_range':50,'curvature':False}):
#
# if bool(boltz['ifinter']):
#     b_data = VasprunBSLoader(vr)
#     print("BSVasprunLoader", type(b_data), b_data)
#     b_inter = BztInterpolator(b_data, lpfac=int(boltz['lpfac']), energy_range=float(boltz['energy_range']),
#                               curvature=bool(boltz['curvature']), save_bztInterp=True,
#                               load_bztInterp=bool(boltz['load']))
#     try:
#         kpath = json.load(open('./kpath', 'r'))
#         kpaths = kpath['path']
#         kpoints_lbls_dict = {}
#         for i in range(len(kpaths)):
#             for j in [0, 1]:
#                 if 'GAMMA' == kpaths[i][j]:
#                     kpaths[i][j] = '\Gamma'
#         for k, v in kpath['kpoints_rel'].items():
#             if k == 'GAMMA':
#                 k = '\Gamma'
#             kpoints_lbls_dict[k] = v
#     except:
#         kpaths = None
#         kpoints_lbls_dict = None
#
#     bs = b_inter.get_band_structure(kpaths=kpaths, kpoints_lbls_dict=kpoints_lbls_dict)
#
#     gap = bs.get_band_gap()
#     nvb = int(np.ceil(nelec / (int(bs.is_spin_polarized) + 1)))
#     vbm = -100
#     print("WHC interpolated gap: %s" % gap)
#     for spin, v in bs.bands.items():
#         vbm = max(vbm, max(v[nvb - 1]))
#     print('WHC WARNNING vasp fermi %s interpolation vbm %s nelec %s nvb %s' % (bs.efermi, vbm, nelec, nvb))
#     if vbm < bs.efermi:
#         bs.efermi = vbm
#
#     if bool(boltz['ifinter']):
#         bs.nvb = nvb
#         bs.ismetaltolerance = float(boltz['ismetaltolerance'])
#
#
# print(bs.bands.keys())
# band_keys = list(bs.bands.keys())
# print(bs.bands[band_keys[0]].shape)
#
# plt.plot(bs.bands[band_keys[0]], label="Fit. Band", color="red", lw=1)
# plt.savefig("Bandstructure Test Combination")
# plt.show()
# plt.close()




def find_vasprun_files():
    """Search for vasprun files from the current directory.

    The precedence order for file locations is:

      1. First search for folders named: 'split-0*'
      2. Else, look in the current directory.

    The split folder names should always be zero based, therefore easily
    sortable.
    """
    folders = glob.glob('split-*')
    folders = sorted(folders) if folders else ['.']

    filenames = []
    for fol in folders:
        vr_file = os.path.join(fol, 'vasprun.xml')
        vr_file_gz = os.path.join(fol, 'vasprun.xml.gz')

        if os.path.exists(vr_file):
            filenames.append(vr_file)
        elif os.path.exists(vr_file_gz):
            filenames.append(vr_file_gz)
        else:
            logging.error('ERROR: No vasprun.xml found in {}!'.format(fol))
            sys.exit()

    return filenames


def save_data_files(bs, prefix=None, directory=None):
    """Write the band structure data files to disk.

    Args:
        bs (`BandStructureSymmLine`): Calculated band structure.
        prefix (`str`, optional): Prefix for data file.
        directory (`str`, optional): Directory in which to save the data.

    Returns:
        The filename of the written data file.
    """
    filename = '{}_band.dat'.format(prefix) if prefix else 'band.dat'
    directory = directory if directory else '.'
    filename = os.path.join(directory, filename)

    if bs.is_metal():
        zero = bs.efermi
    else:
        zero = bs.get_vbm()['energy']

    with open(filename, 'w') as f:
        header = '#k-distance eigenvalue[eV]\n'
        f.write(header)

        # write the spin up eigenvalues
        for band in bs.bands[Spin.up]:
            for d, e in zip(bs.distance, band):
                f.write('{:.8f} {:.8f}\n'.format(d, e - zero))
            f.write('\n')

        # calculation is spin polarised, write spin down bands at end of file
        if bs.is_spin_polarized:
            for band in bs.bands[Spin.down]:
                for d, e in zip(bs.distance, band):
                    f.write('{:.8f} {:.8f}\n'.format(d, e - zero))
                f.write('\n')
    return filename


def _el_orb_tuple(string):
    """Parse the element and orbital argument strings.

    The presence of an element without any orbitals means that we want to plot
    all of its orbitals.

    Args:
        string (`str`): The selected elements and orbitals in in the form:
            `"Sn.s.p,O"`.

    Returns:
        A list of tuples specifying which elements/orbitals to plot. The output
        for the above example would be:

            `[('Sn', ('s', 'p')), 'O']`
    """
    el_orbs = []
    for split in string.split(','):
        splits = split.split('.')
        el = splits[0]
        if len(splits) == 1:
            el_orbs.append(el)
        else:
            el_orbs.append((el, tuple(splits[1:])))
    return el_orbs


def _get_parser():
    parser = argparse.ArgumentParser(description="""
    bandplot is a script to produce publication-ready band
    structure diagrams""",
                                     epilog="""
    Author: {}
    Version: {}
    Last updated: {}""".format(__author__, __version__, __date__))

    parser.add_argument('-f', '--filenames', default=None, nargs='+',
                        metavar='F',
                        help="one or more vasprun.xml files to plot")
    parser.add_argument('-c', '--code', default='vasp',
                        help='Electronic structure code (default: vasp).'
                             '"questaal" also supported.')
    parser.add_argument('-p', '--prefix', metavar='P',
                        help='prefix for the files generated')
    parser.add_argument('-d', '--directory', metavar='D',
                        help='output directory for files')
    parser.add_argument('-b', '--band-edges', dest='band_edges',
                        action='store_true',
                        help='highlight the band edges with markers')
    parser.add_argument('--project', default=None, metavar='S',
                        type=_el_orb_tuple, dest='projection_selection',
                        help=('select which orbitals to project onto the band '
                              'structure (e.g. "Zn.s,Zn.p,O")'))
    parser.add_argument('--mode', default='rgb', type=str,
                        help=('mode for orbital projections (options: rgb, '
                              'stacked)'))
    parser.add_argument('--interpolate-factor', type=int, default=4,
                        dest='interpolate_factor', metavar='N',
                        help=('interpolate factor for band structure '
                              'projections (default: 4)'))
    parser.add_argument('--cartesian', action='store_true',
                        help='Read cartesian k-point coordinates. This is only'
                             ' necessary for some Questaal calculations; Vasp '
                             'outputs are less ambiguous and this option will '
                             'be ignored if --code=vasp.')
    parser.add_argument('--circle-size', type=int, default=150,
                        dest='circle_size', metavar='S',
                        help=('circle size for "stacked" projections '
                              '(default: 150)'))
    parser.add_argument('--ylabel', type=str, default='Energy (eV)',
                        help='y-axis (i.e. energy) label/units')
    parser.add_argument('--dos-label', type=str, dest='dos_label',
                        default=None,
                        help='Axis label for DOS if included')
    parser.add_argument('--dos', default=None,
                        help='path to density of states vasprun.xml')
    parser.add_argument('--elements', type=_el_orb, metavar='E',
                        help='elemental orbitals to plot (e.g. "C.s.p,O")')
    parser.add_argument('--orbitals', type=_el_orb, metavar='O',
                        help=('orbitals to split into lm-decomposed '
                              'contributions (e.g. "Ru.d")'))
    parser.add_argument('--atoms', type=_atoms, metavar='A',
                        help=('atoms to include (e.g. "O.1.2.3,Ru.1.2.3")'))
    parser.add_argument('--spin', type=string_to_spin, default=None,
                        help=('select only one spin channel for a spin-polarised '
                              'calculation (options: up, 1; down, -1)'))
    parser.add_argument('--scissor', type=float, default=None, dest='scissor',
                        help='apply scissor operator')
    parser.add_argument('--total-only', action='store_true', dest='total_only',
                        help='only plot the total density of states')
    parser.add_argument('--no-total', action='store_false', dest='total',
                        help='don\'t plot the total density of states')
    parser.add_argument('--legend-cutoff', type=float, default=3,
                        dest='legend_cutoff', metavar='C',
                        help=('cut-off in %% of total DOS that determines if'
                              ' a line is given a label (default: 3)'))
    parser.add_argument('-g', '--gaussian', type=float, metavar='G',
                        help='standard deviation of DOS gaussian broadening')
    parser.add_argument('--scale', type=float, default=1,
                        help='scaling factor for the density of states')
    parser.add_argument('--height', type=float, default=None,
                        help='height of the graph')
    parser.add_argument('--width', type=float, default=None,
                        help='width of the graph')
    parser.add_argument('--ymin', type=float, default=-6.,
                        help='minimum energy on the y-axis')
    parser.add_argument('--ymax', type=float, default=6.,
                        help='maximum energy on the y-axis')
    parser.add_argument('--style', type=str, nargs='+', default=None,
                        help='matplotlib style specifications')
    parser.add_argument('--no-base-style', action='store_true',
                        dest='no_base_style',
                        help='prevent use of sumo base style')
    parser.add_argument('--config', type=str, default=None,
                        help='colour configuration file')
    parser.add_argument('--format', type=str, default='pdf',
                        dest='image_format', metavar='FORMAT',
                        help='image file format (options: pdf, svg, jpg, png)')
    parser.add_argument('--dpi', type=int, default=400,
                        help='pixel density for image file')
    parser.add_argument('--font', default=None, help='font to use')

    parser.add_argument('--boltz', type=json.loads,
                        default='{"ifinter":"T","nelec":"0","lpfac":"10","energy_range":"50","curvature":"", "soc":"", "load":"", "ismetaltolerance":"0.01"}',
                        help='BoltzTraP parameters')

    parser.add_argument('--nelec', type=int, default=0,
                        help='number of electrons')

    return parser
