#!/usr/bin/env python3
"""
Extract regions from a statistical map.
https://nilearn.github.io/auto_examples/04_manipulating_images/plot_extract_rois_statistical_maps.html
"""
import argparse
import os.path as op

from nilearn.image import threshold_img
from nilearn.regions import connected_regions
from nilearn import plotting
import matplotlib.pyplot as plt

def threshold_map(inputfile, threshold):
    threshold_percentile_img = threshold_img(inputfile, threshold=threshold)
    return threshold_percentile_img

def extract_region(threshold_percentile_img, min_region_size):
    regions_percentile_img, index = connected_regions(threshold_percentile_img,
                                                  min_region_size=min_region_size)
    return regions_percentile_img

def savefig_regions(regions_percentile_img, inputfile, outputfile):
    fig = plt.figure()
    plotting.plot_prob_atlas(regions_percentile_img, bg_img=inputfile,
                        view_type='contours', display_mode='z',
                        cut_coords=10, title='regions', figure=fig)
    plt.savefig(outputfile + '_region_extraction.png')

def save_regions(regions_percentile_img, outputfile):
    regions_percentile_img.to_filename(outputfile + '_region_extraction.nii.gz')

def _cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('infile', type=str,
                        help='Path to 4D dataset')
    parser.add_argument('outfile', type=str,
                        help='Path to outputfile')

    def threshold_type(x):
        x = float(x)
        if not 0 <= x <= 100:
            raise argparse.ArgumentTypeError("Only values between 0 and 100 are accepted.")
        else:
            return str(x) + '%'

    parser.add_argument('--threshold', type=threshold_type, default='97',
                        help='Threshold percentile: 0 < x < 100. Ex. 98')
    
    parser.add_argument('--min_region_size', type=int, default=1500,
                        help='Minimal region size in voxels. Default is 1500')

    return parser


def main(args):
    # -- Check inputs --
    assert op.exists(args.infile), 'Input file not found.'
    assert op.exists(args.outfile), 'Output file not defined.'
    
    # -- Compute and save the regions --
    threshold_percentile_img = threshold_map(args.infile, args.threshold)
    regions_percentile_img = extract_region(threshold_percentile_img, args.min_region_size)
    savefig_regions(regions_percentile_img, args.infile, args.outfile)
    save_regions(regions_percentile_img, args.outfile)

def run_region_extraction():
    parser = _cli_parser()
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    run_region_extraction()
