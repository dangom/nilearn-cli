#!/usr/bin/env python3
"""
Given a 3D or 4D *statistical* image, generate surface plots and output
them as a png file. Uses ImageMagick as a dependency to generate a
mosaic.
"""

import argparse
import multiprocessing
import os
import os.path as op
from functools import partial
from subprocess import call

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from nilearn import datasets, image, plotting, surface
from nilearn._utils.extmath import fast_abs_percentile

FSAVERAGE = datasets.fetch_surf_fsaverage()


def _rename_outfile(nifti):
    """Hackish change of extension from nifti to png.
    Also add some info on the atlas use and the kind of map computed
    """
    if nifti.endswith('nii.gz'):
        name_sans_extension = nifti[:-7]
    elif nifti.endswith('nii'):
        name_sans_extension = nifti[:-4]
    else:
        raise NameError('Cannot recognize nifti extension name.')

    return name_sans_extension + '.png'


def plot_full_surf_stat_map(stat, outname, title=None, **kwargs):
    fig, axes = plt.subplots(dpi=100, nrows=2, ncols=2,
                             subplot_kw={'projection': '3d'})
    ((ax_ll, ax_rl), (ax_lm, ax_rm)) = axes

    # Compute the surface textures from the statistical map.
    texture = surface.vol_to_surf(stat, FSAVERAGE.pial_right)
    texture_l = surface.vol_to_surf(stat, FSAVERAGE.pial_left)

    # Vmax scaled for optimal dynamic range.
    vmax = fast_abs_percentile(stat.dataobj, 99.7)

    plotting.plot_surf_stat_map(FSAVERAGE.infl_right, texture,
                                hemi='right', axes=ax_rl,
                                bg_map=FSAVERAGE.sulc_right,
                                vmax=vmax, **kwargs)
    plotting.plot_surf_stat_map(FSAVERAGE.infl_right, texture,
                                hemi='right', view='medial', axes=ax_rm,
                                bg_map=FSAVERAGE.sulc_right,
                                vmax=vmax, **kwargs)

    plotting.plot_surf_stat_map(FSAVERAGE.infl_left, texture_l,
                                hemi='left', axes=ax_ll,
                                bg_map=FSAVERAGE.sulc_left,
                                vmax=vmax, **kwargs)
    plotting.plot_surf_stat_map(FSAVERAGE.infl_left, texture_l,
                                hemi='left', view='medial', axes=ax_lm,
                                bg_map=FSAVERAGE.sulc_left,
                                vmax=vmax, **kwargs)

    # The default camera distance in the 3D plot makes the surfaces look small.
    # The fix is simple, bring the camera closer to the object.
    for ax in axes.flatten():
        ax.dist = 6
    # Remove whitespace between subplots.
    fig.subplots_adjust(wspace=-0.02, hspace=0.0)
    if title is not None:
        fig.suptitle(title)

    plt.savefig(outname, dpi=100, bbox_inches='tight')


def _plot_wrapper(tup, outdir=None):
    idx, img = tup
    outname = op.join(outdir, f'{idx:02}.png')
    plot_full_surf_stat_map(img, outname, title=f'Volume {idx:02}')


def main(args):
    # -- Check inputs --
    assert op.exists(args.infile), 'Input file not found.'

    if args.outfile is None:
        outfile = _rename_outfile(args.infile)
    else:
        outfile = args.outfile

    if op.exists(outfile):
        raise NameError('Outfile already exists. Not going to overwrite.')

    outdir = op.join(op.dirname(args.infile), 'surface_plot')
    os.makedirs(outdir, exist_ok=True)

    pool = multiprocessing.Pool()
    images = image.iter_img(args.infile)

    pool.map(partial(_plot_wrapper, outdir=outdir), enumerate(images))

    # Non parallel version:
    # for idx, img in enumerate(images):
    #     outname = op.join(outdir, f'{idx:02}.png')
    #     plot_full_surf_stat_map(img, outname, title=f'Volume {idx:02}')

    call(['montage', op.join(outdir, '*.png'),
          '-geometry', '+2+2', outfile])


def _cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile', type=str,
                        help='Path to 4D dataset')
    parser.add_argument('--outfile', type=str, default=None,
                        help=('Name of output file. Default same name '
                              'as file but with png extension'))

    return parser


def run_surface_plot():
    parser = _cli_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_surface_plot()
