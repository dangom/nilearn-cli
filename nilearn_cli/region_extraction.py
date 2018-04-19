#!/usr/bin/env python3
"""
Extract regions from a statistical map.
https://nilearn.github.io/auto_examples/04_manipulating_images/plot_extract_rois_statistical_maps.html
"""
import argparse

from nilearn.image import threshold_img
from nilearn.regions import connected_regions


def extract_region(inputfile, threshold):
    threshold_percentile_img = threshold_img(inputfile, threshold=threshold)
    return threshold_percentile_img


def _cli_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('infile', type=str,
                        help='Path to 4D dataset')

    def threshold_type(x):
        x = float(x)
        if not 0 <= x <= 100:
            raise argparse.ArgumentTypeError("Only values between 0 and 100 are accepted.")
        else:
            return str(x) + '%'

    parser.add_argument('--threshold', type=threshold_type, default='97%',
                        help='Threshold percentile: 0 < x < 100. Ex. 98')

    parser.add_argument('--outfile', type=str, default=None,
                        help=('Name of output file. Default same name '
                              'as file but with png extension'))

    return parser


def main():
    raise NotImplementedError('Not implemented.')


def run_region_extraction():
    parser = _cli_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_region_extraction()
