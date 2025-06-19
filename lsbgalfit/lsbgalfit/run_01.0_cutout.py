#!/usr/bin/env python
"""
Pipeline script for creating image cutouts.
"""
__author__ = "Alex Drlica-Wagner"
import os.path
import logging
import subprocess
import glob

import yaml
import numpy as np

import astropy.io.fits as pyfits

if __name__ == "__main__":
    import parser
    parser = parser.Parser(description=__doc__)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))

    cat = pyfits.open(config['catfile'])[1].data.view(np.recarray)
    objids = cat['COADD_OBJECT_ID']
    if args.objid:
        if np.any(~np.in1d(args.objid,objids)):
            raise Exception("COADD_OBJECT_ID not found")
        objids = np.array(args.objid)

    objids = objids[slice(args.imin,args.imax)]
    for i,objid in enumerate(objids):
        outdir = config['cutdir'].format(objid=objid)
        if not os.path.exists(outdir): os.makedirs(outdir)

        cutfile = os.path.join(outdir,config['cutfile'])
        
        nbands = len(config['bands'])
        nfiles = sum([os.path.exists(cutfile.format(objid=objid,band=b)) for b in config['bands']])
        if nfiles==nbands and not args.force:
            logging.info(f"Found {nbands} cutout files; skipping...")
            continue

        logfile = os.path.join(outdir,config['logfile']).format(objid=objid)
        cmd = f'cutout.py {args.config} -o {objid}'
        if args.force: cmd += ' -f'
        if args.verbose: cmd += ' -v'

        sub = f'csub -q {args.queue} -o {logfile} -n {args.njobs} {cmd}'
        logging.debug(sub)
        subprocess.call(sub,shell=True)

