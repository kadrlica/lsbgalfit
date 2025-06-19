#!/usr/bin/env python
"""
Check tiles on local disk
"""
__author__ = "Alex Drlica-Wagner"
import os.path
import glob
import yaml
import copy
import numpy as np
import logging
import warnings
from astropy.stats import SigmaClip
import matplotlib.pyplot as plt

from scipy import ndimage

from astropy.io import fits as pyfits
from astropy.table import Table,vstack
from astropy.wcs import WCS
#from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
warnings.simplefilter('ignore', UserWarning)
import csv
import pandas as pd
#from astropy.utils.compat.optional_deps import HAS_BOTTLENECK

import galsim
from galsim.des.des_psfex import DES_PSFEx

BANDS = ['g','r','i','z','Y','det']
KEYWORDS = ['TILENAME','BAND','FILTER','MAGZERO','DESFNAME','UNITNAME','ATTNUM','REQNUM']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('-b','--band',action='append',choices=BANDS,
                       help='choose a band')
    parser.add_argument('-f','--force',action='store_true',
                       help='force overwrite')
    parser.add_argument('-v','--verbose',action='store_true',
                       help='output verbosity')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(level)
    
    config = yaml.safe_load(open(args.config))
    bands = args.band if args.band else config['bands']

    cat = pyfits.open(config['catfile'])[1].data.view(np.recarray)
    tiles = np.unique(cat['TILENAME'].astype(str))
    
    
    bad = Table(names=['TILENAME','IMFLAG','PSFFLAG','SEGFLAG'],dtype=[str,int,int,int])
    
    bad_files = []
    for tile in tiles:
        tilepath = os.path.join(config['datadir'],tile)
        im_flag = 0
        psf_flag = 0
        seg_flag = 0
        for i,band in enumerate(BANDS):
            logging.debug(f'Checking tile: {tile}')
            # Pick up fits or fits.fz
            imgfile = glob.glob(tilepath+f'/*_{band}.fits.fz')[0]
            psffile = glob.glob(tilepath+f'/*_{band}_psfcat.psf')[0]
            segfile = glob.glob(tilepath+f'/*_{band}_segmap*')[0]
            
            logging.debug(f'Checking image file: {tile},{band}')
            try:
                img = pyfits.open(imgfile)
            except:
                logging.debug('Image corrupted. Logging error.')
                bad_files.append(imgfile)
                flag = 1 << i
                im_flag = im_flag | flag
            try:
                psf = DES_PSFEx(psffile)
            except:
                logging.debug('PSF corrupted. Logging error.')
                bad_files.append(psffile)
                flag = 1 << i
                psf_flag = psf_flag | flag
                
            try:
                seg = pyfits.open(segfile)
            except:
                logging.debug('Segfile corrupted. Logging error.')
                bad_files.append(segfile)
                flag = 1 << i
                seg_flag = seg_flag | flag
                
        if im_flag != 0 or psf_flag != 0:
            arr = [tile,im_flag,psf_flag,seg_flag]
            bad.add_row(arr)
    
    logging.debug('Writing bad tile file...')
    bad.write('y6_gold_2_0_bad_tiles.fits',format='fits',overwrite=True)
    
    with open('bad_files.csv','w') as file:
        write = csv.writer(file)
        write.writerows(bad_files)
        
    bad_dict = {'FILES': bad_files}
    df = pd.DataFrame(bad_dict)
    df.to_csv('bad_files_backup.csv')
                          
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
