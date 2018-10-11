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
from nilearn.input_data import (NiftiLabelsMasker, NiftiSpheresMasker,
                                NiftiMapsMasker)


# Voxels with 0 intensity correspond to the background.
YEO_LABELS = ['Background', 'Visual', 'Somato-sensor',
              'Dorsal Attention', 'Ventral Attention',
              'Limbic', 'Frontoparietal', 'Default']


ATLAS_3D = dict(
    cortical=(lambda:
              datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')),
    subcortical=(lambda:
                 datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')),
    yeo7=(lambda:
          dict(maps=datasets.fetch_atlas_yeo_2011()['thick_7'],
               labels=YEO_LABELS)),
    aal=(lambda:
         _fetch_aal()),
    # brodmann=(lambda:
    #           datasets.fetch_atlas_talairach('ba'))
)

ATLAS_COORDS = dict(
    power=(lambda:
           datasets.fetch_coords_power_2011()),
)

ATLAS_PROBABILISTIC = dict(
    msdl=(lambda:
          _fetch_msdl()))


def _fetch_aal():
    """ The AAL atlas does not contain a background label.
    To make the API consistent we fix it here.
    """
    aal = datasets.fetch_atlas_aal()
    aal['labels'] = ['Background'] + aal['labels']
    return aal


def _fetch_msdl():
    """ The AAL atlas does not contain a background label.
    To make the API consistent we fix it here.
    """
    msdl = datasets.fetch_atlas_msdl()
    msdl['labels'] = ['Background'] + msdl['labels']
    return msdl


def _rename_outfile(nifti, atlas, kind='correlation', confounds=False):
    """Hackish change of extension from nifti to png.
    Also add some info on the atlas use and the kind of map computed
    """
    if nifti.endswith('nii.gz'):
        name_sans_extension = nifti[:-7]
    elif nifti.endswith('nii'):
        name_sans_extension = nifti[:-4]
    else:
        raise NameError('Cannot recognize nifti extension name.')

    name = name_sans_extension + '_' + atlas + '_' + kind
    if confounds:
        name += '_wconfounds'
    return name + '.png'


def fetch_atlas(name):
    """
    This function isn't doing much right now. In the case
    I ever expand the atlas possibilities, than a proper
    filename, label logic can go in here so that the function always
    returns a tuple (atlas_filename, atlas_labels)
    """
    dataset = ATLAS_3D[name]()
    return dataset['maps'], dataset['labels']


def fetch_coords(name):
    """
    This function isn't doing much right now. In the case
    I ever expand the atlas possibilities, than a proper
    filename, label logic can go in here so that the function always
    returns a tuple (atlas_filename, atlas_labels)
    """
    dataset = ATLAS_COORDS[name]()
    labels = dataset['labels'] if 'labels' in dataset.keys() else None
    return dataset['rois'], labels


def fetch_probabilistic(name):
    """
    This function isn't doing much right now. In the case
    I ever expand the atlas possibilities, than a proper
    filename, label logic can go in here so that the function always
    returns a tuple (atlas_filename, atlas_labels)
    """
    dataset = ATLAS_PROBABILISTIC[name]()
    labels = dataset['labels'] if 'labels' in dataset.keys() else None
    return dataset['maps'], labels


def extract_timeseries(filename, atlas_filename, confounds=None):
    """
    Wrapper around nilearn masker and fit_transform.
    """
    masker = NiftiLabelsMasker(labels_img=atlas_filename,
                               standardize=True)

    time_series = masker.fit_transform(filename,
                                       confounds=confounds)
    return time_series


def extract_timeseries_coords(filename, raw_coords, confounds=None):
    """Because the power parcellation is given in coordinates and not labels,
    we dedicate an exclusive function to deal with it.
    """
    coords = np.vstack((raw_coords.rois['x'],
                        raw_coords.rois['y'],
                        raw_coords.rois['z'])).T

    spheres_masker = NiftiSpheresMasker(
        seeds=coords, radius=5., standardize=True)

    time_series = spheres_masker.fit_transform(filename,
                                               confounds=confounds)
    return time_series


def extract_timeseries_probabilistic(filename, maps, confounds=None):
    """Because the power parcellation is given in coordinates and not labels,
    we dedicate an exclusive function to deal with it.
    """
    maps_masker = NiftiMapsMasker(
        maps, resampling_target="data", standardize=True)

    time_series = maps_masker.fit_transform(filename,
                                            confounds=confounds)
    return time_series


def generate_connectivity_matrix(time_series, kind='correlation'):
    """
    Generate a connectivity matrix from a collection of time series.
    param :kind: Any kind accepted by nilearn ConnectivityMeasure
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
    # -- Check inputs --
    assert op.exists(args.infile), 'Input file not found.'

    if args.confounds is not None:
        assert op.exists(args.confounds), 'Confounds file not found.'
        confounds = args.confounds
    elif args.fmriprepconfounds:
        confounds = args.infile[:args.infile.find('bold')+4] + '_confounds.tsv'
        confounds = np.nan_to_num(np.genfromtxt(confounds,
                                                missing_values=('n\a'),
                                                skip_header=1))
    else:
        confounds = None

    if args.outfile is None:
        conf = True if confounds is not None else False
        outfile = _rename_outfile(args.infile, args.atlas, args.kind, conf)
    else:
        outfile = args.outfile

    # TODO implement proper sanity check:
    if op.exists(outfile):
        raise NameError('Outfile already exists. Not going to overwrite.')

    # -- Compute and save the connectome --
    if args.atlas in ATLAS_COORDS.keys():
        atlas_rois, atlas_labels = fetch_coords(args.atlas)
        ts = extract_timeseries_coords(args.infile, atlas_rois, confounds)
    elif args.atlas in ATLAS_PROBABILISTIC.keys():
        maps, atlas_labels = fetch_probabilistic(args.atlas)
        ts = extract_timeseries_probabilistic(args.infile, maps,
                                              confounds)
    else:
        atlas_filename, atlas_labels = fetch_atlas(args.atlas)
        ts = extract_timeseries(args.infile, atlas_filename, confounds)

    kind = args.kind if (
        args.kind != 'partialcorrelation') else 'partial correlation'
    connectivity_matrix = generate_connectivity_matrix(ts, kind)
    if args.savetxt:
        outfiletxt = op.splitext(outfile)[0] + '.txt'
        np.savetxt(outfiletxt, connectivity_matrix, fmt='%f')
    savefig_connectome(connectivity_matrix, outfile, title=args.title,
                       labels=atlas_labels, reorder=args.no_reorder)


def _cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('infile', type=str,
                        help='Path to 4D dataset')

    atlas_choices = [*ATLAS_PROBABILISTIC.keys(),
                     *ATLAS_3D.keys(),
                     *ATLAS_COORDS.keys()]

    parser.add_argument('--atlas', type=str, default='cortical',
                        choices=atlas_choices,
                        help=('Which atlas to use? Default cortical'))

    parser.add_argument('--confounds', default=None,
                        help='Path to tsv or csv file of confounds.')

    parser.add_argument('--fmriprepconfounds', action='store_true',
                        help='Autodetect fmriprep confound file')

    parser.add_argument('--title', default=None,
                        help='Add a title to the figure')

    parser.add_argument('--kind', default='correlation',
                        choices=['correlation',
                                 'partialcorrelation',
                                 'partial correlation',
                                 'tangent',
                                 'covariance',
                                 'precision'],
                        help='Kind of connectivity measure to compute')

    parser.add_argument('--outfile', type=str, default=None,
                        help=('Name of output file. Default same name '
                              'as file but with png extension'))

    parser.add_argument('--savetxt', action='store_true',
                        help=('Same matrix as txt as well.'))

    parser.add_argument('--no-reorder', action='store_false',
                        help=('Reorder connectome plot.'))

    return parser


def run_connectome():
    parser = _cli_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_connectome()
