#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os, sys
from os.path import join

import pandas as pd
import numpy as np
import subprocess

DTYPES = ['coadd','seg','psf']
#BANDS = ['g','r','i','det']
BANDS = ['z','Y']

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d','--dtype',action='append',choices=DTYPES,default=None)
parser.add_argument('-b','--band',action='append',choices=BANDS,default=None)
args = parser.parse_args()

if args.band is None:
    args.band = ['i']
if args.dtype is None:
    args.dtype = ['seg']

outdir = 'image'
urlpath = 'image-urls/v1'

# Old CSV files include ALHAMBRA fields
#print("WARNING: Files need to be updated")
#sys.exit()

coadd_seg = pd.read_csv(join(urlpath,'coadd_seg.csv')).to_records(index=False)
coadd_img = pd.read_csv(join(urlpath,'coadd_img.csv')).to_records(index=False)
coadd_psf = pd.read_csv(join(urlpath,'coadd_psf.csv')).to_records(index=False)

print("Downloading dtype: %s"%args.dtype)
print("Downloading bands: %s"%args.band)

def download_image(d,njobs=20):
    outdir = 'image/%(TILENAME)s'%d
    outbase = d['URL'].rsplit('/')[-1]
    outfile = join(outdir,outbase)
    if os.path.exists(outfile):
        print("Found %s; skipping..."%(outfile))
        return

    cmd = 'wget %s -O %s'%(d['URL'],outfile)
    sub = 'csub -q vanilla -n %s %s'%(njobs,cmd)
    print(sub)
    return subprocess.check_call(sub,shell=True)
    
if 'seg' in args.dtype:
    data = coadd_seg
    data = data[np.in1d(data['BAND'],args.band)]
    for d in data:
        download_image(d)

if 'coadd' in args.dtype:
    data = coadd_img
    data = data[np.in1d(data['BAND'],args.band)]
    for d in data:
        download_image(d)

if 'psf' in args.dtype:
    data = coadd_psf
    data = data[np.in1d(data['BAND'],args.band)]
    for d in data:
        download_image(d,njobs=30)
