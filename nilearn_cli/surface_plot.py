#!/usr/bin/env python3
"""
Given a 3D or 4D *statistical* image, generate surface plots and output
them as a png file. Uses ImageMagick as a dependency to generate a
mosaic in the case of 4D images.
"""

import argparse
import multiprocessing
import os.path as op
import tempfile
from functools import partial
from subprocess import call

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from nilearn import datasets, image, plotting, surface
from nilearn._utils.extmath import fast_abs_percentile

FSAVERAGE = datasets.fetch_surf_fsaverage()


def hcp_cmap(name='roy_big_bl'):

    # Original colormap definition:
    # https://github.com/Washington-University/workbench/blob/bleeding/src/Files/PaletteFile.cxx#L1178

    cmaps = {

        # Written in reversed order, that's why [::-1]
        'roy_big_bl': (np.array([(255, 255, 0), (255, 200, 0),
                                 (255, 120, 0), (255, 0, 0),
                                 (200, 0, 0), (150, 0, 0),
                                 (100, 0, 0), (60, 0, 0),
                                 (0, 0, 0), (0, 0, 80),
                                 (0, 0, 170), (75, 0, 125),
                                 (125, 0, 160), (75, 125, 0),
                                 (0, 200, 0), (0, 255, 0),
                                 (0, 255, 255),
                                 (0, 255, 255)][::-1]) / 255).tolist(),

        'videen_style': ['#000000', '#bbbbbb', '#dddddd', '#ffffff',
                         '#ff388d', '#e251e2', '#10b010', '#00ff00',
                         '#00ffff', '#000000', '#660033', '#33334c',
                         '#4c4c7f', '#7f7fcc', '#00ff00', '#10b010',
                         '#ffff00', '#ff9900', '#ff6900', '#ff0000']
    }

    return mpl.colors.LinearSegmentedColormap.from_list(
        name, cmaps[name])


def _rename_outfile(nifti):
    """Hackish change of extension from nifti to png.
    """
    if nifti.endswith('nii.gz'):
        name_sans_extension = nifti[:-7]
    elif nifti.endswith('nii'):
        name_sans_extension = nifti[:-4]
    else:
        raise NameError('Cannot recognize nifti extension name.')

    return name_sans_extension + '.png'


def plot_full_surf_stat_map(stat, title=None, ts=None, mask=None,
                            inflate=False, outfile=None,
                            vmax=None, **kwargs):
    """Use nilearn's plot_surf_stat_map to plot volume data in the surface.
    Plots a collage of both hemispheres and both medial and lateral views of
    the brain. The surface mesh used for plotting is freesurfer's fsaverage.

    :param stat: A 3D statistical map (Nilearn's niimg object)
    :param title: A title for the image
    :param ts: Optional timeseries to be plotted in the background.
    :param mask: Optional mask passed to nilearn's vol_to_surf.
    :param inflate: If True, plot on inflated, if False, on pial.
    :param outfile: Optional output filename to save figure.
    :param vmax: Colormap vmax limit

    Other kwargs are passed to Nilearn's plot_surf_stat_map.
    """

    fig, axes = plt.subplots(dpi=100, nrows=2, ncols=2,
                             subplot_kw={'projection': '3d'})
    ((ax_ll, ax_rl), (ax_lm, ax_rm)) = axes

    # Compute the surface textures from the statistical map.
    texture_r = surface.vol_to_surf(stat, FSAVERAGE.pial_right,
                                    mask_img=mask)
    texture_l = surface.vol_to_surf(stat, FSAVERAGE.pial_left,
                                    mask_img=mask)

    # Vmax scaled for optimal dynamic range.
    if vmax is None:
        vmax = fast_abs_percentile(stat.dataobj, 99.8)

    # Plot on inflated brain or on pial surface?
    if inflate:
        leftsurf = FSAVERAGE.infl_left
        rightsurf = FSAVERAGE.infl_right
    else:
        leftsurf = FSAVERAGE.pial_left
        rightsurf = FSAVERAGE.pial_right

    plotting.plot_surf_stat_map(rightsurf, texture_r,
                                hemi='right', axes=ax_rl,
                                bg_map=FSAVERAGE.sulc_right,
                                vmax=vmax, **kwargs)
    plotting.plot_surf_stat_map(rightsurf, texture_r,
                                hemi='right', view='medial', axes=ax_rm,
                                bg_map=FSAVERAGE.sulc_right,
                                vmax=vmax, **kwargs)

    plotting.plot_surf_stat_map(leftsurf, texture_l,
                                hemi='left', axes=ax_ll,
                                bg_map=FSAVERAGE.sulc_left,
                                vmax=vmax, **kwargs)
    plotting.plot_surf_stat_map(leftsurf, texture_l,
                                hemi='left', view='medial', axes=ax_lm,
                                bg_map=FSAVERAGE.sulc_left,
                                vmax=vmax, **kwargs)

    # The default camera distance in the 3D plot makes the surfaces look small.
    # The fix is simple, bring the camera closer to the object.
    for ax in axes.flatten():
        ax.dist = 6
        # Alpha set to 0 so that the timeseries is visible.
        ax.patch.set_alpha(0.)
    # Remove whitespace between subplots.
    fig.subplots_adjust(wspace=-0.02, hspace=0.0)

    if ts is not None:
        # x0, y0, width, height = 0.34, 0.465, 0.38, 0.06
        # x0 left -> right, y0 top -> down.
        x0, y0 = 0.15, 0.445
        width, height = (1.05 - 2*x0), 0.12
        ax5 = fig.add_axes([x0, y0, width, height], zorder=-1)
        ax5.plot(ts[::2], 'dimgray', linewidth=0.8)
        ax5.axis('off')
        # Only necessary if ax5 is on top. (zorder larger than other axes)
        # ax5.patch.set_alpha(0.)

    if title is not None:
        # y defaults to 0.98. The value of 0.93 lowers it a bit.
        fig.suptitle(title, fontsize=12, y=0.93)

    if outfile is not None:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')


def _plot_wrapper(tup, outdir=None, threshold=None, mask=None, label='Volume',
                  tsfile=None, inflate=False, **kwargs):
    """Small wrapper at the module level compatible with pool.map()
    to call multiple instances of plot_full_surf_stat_map in parallel.
    """
    idx, img = tup
    outfile = op.join(outdir, f'{idx:04}.png')
    ts = np.loadtxt(tsfile)[:, idx] if tsfile is not None else None
    plot_full_surf_stat_map(img, outfile=outfile,
                            ts=ts, title=f'{label} {(idx+1):02}',
                            threshold=threshold, mask=mask, inflate=inflate,
                            **kwargs)


def main(args):
    # -- Check inputs --
    assert op.exists(args.infile), 'Input file not found.'
    if args.mask is not None:
        assert op.exists(args.mask), 'Mask file not found.'

    if args.outfile is None:
        outfile = _rename_outfile(args.infile)
    else:
        outfile = args.outfile

    cmap = hcp_cmap() if args.deceive else 'cold_hot'

    # TODO simplify if else to avoid duplication.
    if len(image.load_img(args.infile).shape) < 4:  # Handle 3D image
        img = image.load_img(args.infile)
        plot_full_surf_stat_map(img, outfile=outfile, title=args.label,
                                cmap=cmap,
                                bg_on_data=args.bg_on_data, vmax=args.vmax,
                                threshold=args.threshold,
                                inflate=args.inflate, mask=args.mask)

    else:  # Handle 4D images.
        originaldir = op.dirname(args.infile)
        tmpdir = tempfile.mkdtemp()

        pool = multiprocessing.Pool()
        images = image.iter_img(args.infile)

        if op.exists(op.join(originaldir, 'melodic_mix')):
            tsfile = op.join(originaldir, 'melodic_mix')
        else:
            tsfile = None

        pool.map(partial(_plot_wrapper, outdir=tmpdir, cmap=cmap,
                         label=args.label,
                         bg_on_data=args.bg_on_data, vmax=args.vmax,
                         threshold=args.threshold, tsfile=tsfile,
                         inflate=args.inflate, mask=args.mask),
                 enumerate(images))

        # Use ImageMagick's montage to create a mosaic of all individual plots.
        call(['montage', op.join(tmpdir, '*.png'), '-trim',
              '-geometry', '+9+9', outfile])


def _cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile', type=str,
                        help='Path to 4D dataset')
    parser.add_argument('--outfile', type=str, default=None,
                        help=('Name of output file. Default same name '
                              'as file but with png extension'))
    parser.add_argument('--label', type=str, default='Volume projected on surface',
                        help=('Label for individual plots'))
    parser.add_argument('--threshold', type=float, default=None,
                        help=('Value to (lower) threshold maps'))
    parser.add_argument('--vmax', type=float, default=None,
                        help=('Set colorbar limit'))
    parser.add_argument('--inflate', action='store_true',
                        help=('Instead of plotting on pial surface,'
                              'plot on inflated brain'))
    parser.add_argument('--bg-on-data', action='store_true',
                        help=('Mix background image with statistical image'
                              'Modifies stats according to sulcus depth.'))
    parser.add_argument('--mask', type=str, default=None,
                        help=('Mask to compute volume to surface.'))
    parser.add_argument('--deceive', action='store_true',
                        help=(('Use HCP colormap. Warning: Non-perceptually uniform.'
                               'Colormap may generate clusters that do not really exist.')))

    return parser


def run_surface_plot():
    parser = _cli_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_surface_plot()
