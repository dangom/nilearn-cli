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
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.input_data import NiftiLabelsMasker

ATLAS = dict(
    cortical=(lambda:
              fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')),
    subcortical=(lambda:
                 fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm'))
)


def _change_extension(nifti):
    """Hackish change of extension from nifti to png.
    """
    if nifti.endswith('nii.gz'):
        return nifti[:-7] + '.png'
    elif nifti.endswith('nii'):
        return nifti[:-4] + '.png'
    else:
        raise NameError('Cannot recognize nifti extension name.')


def fetch_atlas(name):
    """
    This function isn't doing much right now. In the case
    I ever expand the atlas possibilities, than a proper
    filename, label logic can go in here so that the function always
    returns a tuple (atlas_filename, atlas_labels)
    """
    return ATLAS[name]()


def extract_timeseries(filename, atlas_filename, confounds=None):
    """
    Wrapper around nilearn masker and fit_transform.
    """
    masker = NiftiLabelsMasker(labels_img=atlas_filename,
                               standardize=True)

    time_series = masker.fit_transform(filename,
                                       confounds=confounds)
    return time_series


def generate_connectivity_matrix(time_series, kind='correlation'):
    """
    Generate a connectivity matrix from a collection of time series.
    """
    connectivity_measure = ConnectivityMeasure(kind=kind)
    connectivity_matrix = connectivity_measure.fit_transform([time_series])[0]
    return connectivity_matrix


def savefig_connectome(connectivity_matrix, atlas_labels, outfile, title=None):
    """
    Save a connectome figure using nilearn's plot_matrix.
    """
    connectome = connectivity_matrix.copy()
    np.fill_diagonal(connectome, 0)
    fig = plt.figure(figsize=(10, 8))
    plotting.plot_matrix(connectome,
                         figure=fig,
                         labels=atlas_labels[1:],
                         vmin=-0.8,
                         vmax=0.8,
                         reorder=True)

    if title is not None:
        fig.suptitle(title)

    plt.savefig(outfile)


def main(args):
    """Main connectome routine.
    """
    assert op.exists(args.infile)

    if args.confounds is not None:
        assert op.exists(args.confounds)

    atlas_filename, atlas_labels = fetch_atlas(args.atlas)
    ts = extract_timeseries(args.infile, atlas_filename, args.confounds)
    connectivity_matrix = generate_connectivity_matrix(ts, args.kind)
    if args.outfile is None:
        outfile = _change_extension(args.infile, '.png')

    savefig_connectome(connectivity_matrix, atlas_labels, outfile)


def _cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('infile', type=str,
                        help='Path to 4D dataset')

    parser.add_argument('atlas', type=str, choices=['cortical', 'subcortical'],
                        help='Harvard-Oxford atlas')

    parser.add_argument('confounds', default=None,
                        help='Path to tsv or csv file of confounds.')

    parser.add_argument('kind', default='correlation',
                        choices=['correlation',
                                 'partial correlation',
                                 'tangent',
                                 'covariance',
                                 'precision'],
                        help='Kind of connectivity measure to compute')

    parser.add_argument('outfile', type=str, default=None,
                        help=('Name of output file. Default same name '
                              'as file but with png extension'))

    return parser


def run_connectome():
    parser = _cli_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_connectome()
