#!/usr/bin/env python3
"""
Extract signals from a brain parcellation as demonstrated in
https://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html
"""
import argparse
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
import nilearn.datasets as datasets
from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker


# Voxels with 0 intensity correspond to the background.
YEO_LABELS = ['Background', 'Visual', 'Somato-sensor',
              'Dorsal Attention', 'Ventral Attention',
              'Limbic', 'Frontoparietal', 'Default']

ATLAS = dict(
    cortical=(lambda:
              datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')),
    subcortical=(lambda:
                 datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')),
    yeo7=(lambda:
          dict(maps=datasets.fetch_atlas_yeo_2011()['thick_7'],
               labels=YEO_LABELS))
)


def _rename_outfile(nifti, atlas, kind='correlation'):
    """Hackish change of extension from nifti to png.
    """
    if nifti.endswith('nii.gz'):
        name_sans_extension = nifti[:-7]
    elif nifti.endswith('nii'):
        name_sans_extension = nifti[:-4]
    else:
        raise NameError('Cannot recognize nifti extension name.')

    return name_sans_extension + '_' + atlas + '_' + kind + '.png'


def fetch_atlas(name):
    """
    This function isn't doing much right now. In the case
    I ever expand the atlas possibilities, than a proper
    filename, label logic can go in here so that the function always
    returns a tuple (atlas_filename, atlas_labels)
    """
    dataset = ATLAS[name]()
    return dataset['maps'], dataset['labels']


def extract_timeseries(filename, atlas_filename, confounds=None):
    """
    Wrapper around nilearn masker and fit_transform.
    """
    masker = NiftiLabelsMasker(labels_img=atlas_filename,
                               standardize=True)

    time_series = masker.fit_transform(filename,
                                       confounds=confounds)
    return time_series


def extract_timeseries_power(filename, confounds=None):
    """Because the power parcellation is given in coordinates and not labels,
    we dedicate an exclusive function to deal with it.
    """
    power = datasets.fetch_coords_power_2011()
    coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T

    spheres_masker = NiftiSpheresMasker(
        seeds=coords, radius=5., standardize=True)

    timeseries = spheres_masker.fit_transform(filename,
                                              confounds=confounds)
    return timeseries


def generate_connectivity_matrix(time_series, kind='correlation'):
    """
    Generate a connectivity matrix from a collection of time series.
    """
    connectivity_measure = ConnectivityMeasure(kind=kind)
    connectivity_matrix = connectivity_measure.fit_transform([time_series])[0]
    return connectivity_matrix


def savefig_connectome(connectivity_matrix, outfile, labels=None,
                       title=None, reorder=True):
    """
    Save a connectome figure using nilearn's plot_matrix.
    """
    fake_labels = [str(i) for i in range(1, connectivity_matrix.shape[0] + 1)]
    atlas_labels = labels[1:] if labels is not None else fake_labels

    connectome = connectivity_matrix.copy()
    np.fill_diagonal(connectome, 0)

    fig = plt.figure(figsize=(10, 8))

    plotting.plot_matrix(connectome,
                         figure=fig,
                         labels=atlas_labels,
                         vmin=-0.8,
                         vmax=0.8,
                         reorder=reorder)

    if title is not None:
        fig.suptitle(title)

    plt.savefig(outfile)


def main(args):
    """Main connectome routine.
    """
    assert op.exists(args.infile)

    if args.confounds is not None:
        assert op.exists(args.confounds)

    if args.atlas == 'power':
        atlas_labels = None
        ts = extract_timeseries_power(args.infile, args.confounds)
    else:
        atlas_filename, atlas_labels = fetch_atlas(args.atlas)
        ts = extract_timeseries(args.infile, atlas_filename, args.confounds)

    connectivity_matrix = generate_connectivity_matrix(ts, args.kind)

    if args.outfile is None:
        outfile = _rename_outfile(args.infile, args.atlas, args.kind)

    # TODO implement proper sanity check:
    if op.exists(outfile):
        raise NameError('Outfile already exists. Not going to overwrite.')

    savefig_connectome(connectivity_matrix, outfile,
                       labels=atlas_labels, reorder=args.no_reorder)


def _cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('infile', type=str,
                        help='Path to 4D dataset')

    parser.add_argument('--atlas', type=str, default='power',
                        choices=['power', 'yeo7', 'cortical', 'subcortical'],
                        help='Which atlas to use? Default power')

    parser.add_argument('--confounds', default=None,
                        help='Path to tsv or csv file of confounds.')

    parser.add_argument('--kind', default='correlation',
                        choices=['correlation',
                                 'partial correlation',
                                 'tangent',
                                 'covariance',
                                 'precision'],
                        help='Kind of connectivity measure to compute')

    parser.add_argument('--outfile', type=str, default=None,
                        help=('Name of output file. Default same name '
                              'as file but with png extension'))

    parser.add_argument('--no-reorder', action='store_false',
                        help=('Reorder connectome plot.'))

    return parser


def run_connectome():
    parser = _cli_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_connectome()
