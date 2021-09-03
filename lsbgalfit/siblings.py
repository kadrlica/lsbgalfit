#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import os.path
import glob
import yaml
import subprocess

import numpy as np
import logging

from astropy.io import fits as pyfits
url="https://data.darkenergysurvey.org/fnalmisc/lsbgalfit/v3/"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('-o','--objid',nargs='+',
                        type=int,
                        help='cutouts by coadd_object_id')
    parser.add_argument('-c','--csv',type=str,
                        help='cutouts by coadd_object_id')
    parser.add_argument('-v','--verbose',action='store_true',
                       help='output verbosity')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(level)

    config = yaml.safe_load(open(args.config))

    group = pyfits.open(config['grpfile'])[1].data.view(np.recarray)

    if args.csv:
        print("Reading csv file: {args.csv}")
        csv = np.genfromtxt(args.csv,delimiter=',',skip_header=1,dtype=None)
        sel = (csv[csv.dtype.names[1]] == 'seg')
        args.objid = csv[0][sel]

    print("Found {len(args.objid)} objects...")

    for objid in args.objid:
        grp = group[group['LSBG_COADD_OBJECT_ID'] == objid]
        print(f'{objid}:')
        print(f"  group: {list(grp['COADD_OBJECT_ID'])}")
        print()
        #print(url+f'segmap_{objid}.png')
        #path = f"../v3/galfit-v4/{objid}"
        path = f"../v6/galfit-v0/{objid}"
        cmd = f"display {path}/segmap_{objid}.png {path}/results_{objid}.png"
        subprocess.call(cmd,shell=True)
