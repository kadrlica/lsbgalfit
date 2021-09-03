#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import os.path
import shutil
import logging
import subprocess
import glob
import yaml

import numpy as np

from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

def build_galfit(config,obj):
    """Build galfitm configuration files"""
    build_feedme(config,obj)
    build_constraint(config,obj)

def build_constraint(config, obj):
    """Build a constraints file for the specified object"""
    objid = obj['COADD_OBJECT_ID']
    logging.info(objid)
    bands = config['bands']

    # Load default constraints
    params = config['constraint']

    # Defaults
    params.setdefault('skymin_const',0)
    params.setdefault('skymax_const',0)

    # Update constraints on centroid
    reff  = np.max([obj[f'FLUX_RADIUS_{band.upper()}'] for band in bands])
    params.update(x_const=max(params['x_const'],reff/5.)) #pixels
    params.update(y_const=max(params['y_const'],reff/5.)) #pixels

    # Update custom constraints
    custom=yaml.safe_load(open(os.path.join(config['tempdir'],config['custfile'])))
    try: params.update(custom[objid]['constraint'])
    except KeyError: pass

    # construct feedme
    filename = os.path.join(config['tempdir'],config['constfile'])
    with open(filename, 'r') as myfile:
        constraint = myfile.read()

    constraint = constraint.format(**params)

    outdir = config['galdir'].format(objid=objid)
    outfile = os.path.join(outdir,config['constfile'])
    logging.info(f"Writing {outfile}...")
    with open(outfile, 'w') as myfile:
        myfile.write(constraint)
        myfile.close()

    return outfile

def build_feedme(config, obj):
    """Build a feedme file for the specified object"""
    objid = obj['COADD_OBJECT_ID']
    logging.info(objid)

    bands = config['bands']
    basedir = os.getcwd()
    cutdir = os.path.join(basedir,config['cutdir']).format(objid=objid)
    outdir = config['galdir'].format(objid=objid)

    imgblock = config['galfile'].format(objid=objid)
    
    cutfile = os.path.join(cutdir,config['cutfile'])
    psffile = os.path.join(cutdir,config['psffile'])

    imgpath = ','.join(cutfile.format(objid=objid,band=band)+'[SCI]' for band in bands)
    mskpath = ','.join(cutfile.format(objid=objid,band=band)+'[BAD]' for band in bands)
    sigpath = ','.join(cutfile.format(objid=objid,band=band)+'[SIG]' for band in bands)
    psfpath = ','.join(cutfile.format(objid=objid,band=band)+'[PSF]' for band in bands)
    #psfpath = ','.join(psffile.format(objid=objid,band=band) for band in bands)
    bandout = ','.join(bands)
    magstr  = ','.join('%.2f'%obj[f'MAG_AUTO_{band.upper()}'] for band in bands)

    #reffout = ','.join('%.2f'%obj[f'FLUX_RADIUS_{band.upper()}'] for band in bands)
    reff    = np.max([obj[f'FLUX_RADIUS_{band.upper()}'] for band in bands])
    reffstr = '%.2f'%reff
    ellip   = obj['B_IMAGE']/obj['A_IMAGE']
    pa      = obj['THETA_J2000']

    hdr = pyfits.open(cutfile.format(objid=objid,band=bands[0]))['SCI'].header
    xmax,ymax = hdr['NAXIS1'],hdr['NAXIS2']
    #xmax,ymax = config['size'],config['size']

    # construct feedme
    feedfile = os.path.join(config['tempdir'],config['feedfile'])
    with open(feedfile, 'r') as myfile:
        feedme = myfile.read()

    feedme = feedme.format(config=config,
                           imgpath=imgpath,
                           sigpath=sigpath,
                           psfpath=psfpath,  
                           mskpath=mskpath,
                           outpath=imgblock, 
                           constraints=config['constfile'],
                           bands=bandout,
                           xmax=xmax, ymax=ymax,
                           posx=xmax/2., posy=ymax/2.,
                           magstr=magstr, reffstr=reffstr,
                           ellip=ellip, pa=pa)

    # save feedme to disk
    if not os.path.exists(outdir): os.makedirs(outdir)

    outfile = os.path.join(outdir,config['feedfile'])
    logging.info(f"Writing {outfile}...")
    with open(outfile, 'w') as myfile:
        myfile.write(feedme)
        myfile.close()

    return outfile

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('-o','--objid',nargs='+',action='append',
                        type=int,required=True,
                        help='cutouts by coadd_object_id')
    parser.add_argument('-f','--force',action='store_true',
                       help='force overwrite')
    parser.add_argument('-v','--verbose',action='store_true',
                       help='output verbosity')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(level)

    config = yaml.safe_load(open(args.config))
    cat = pyfits.open(config['catfile'])[1].data.view(np.recarray)
    objids = np.array(args.objid)

    print("Generating feedmes for %i objects..."%len(objids))
    objects = cat[np.in1d(cat['COADD_OBJECT_ID'],objids)]

    for i,obj in enumerate(objects):
        build_galfit(config,obj)
