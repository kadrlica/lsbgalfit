#!/usr/bin/env python
"""
Pipeline script for creating galfit feedme.
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

    cat = pyfits.open(config['catfile'])[1].data.view(np.recarray)
    objids = cat['COADD_OBJECT_ID']
    if args.objid:
        if np.any(~np.in1d(args.objid,objids)):
            raise Exception("COADD_OBJECT_ID not found")
        objids = np.array(args.objid)

    objids = objids[slice(args.imin,args.imax)]
    for i,objid in enumerate(objids):
        outdir = config['galdir'].format(objid=objid)
        if not os.path.exists(outdir): os.makedirs(outdir)

        outfile = os.path.join(outdir,'*.feedme')
        if len(glob.glob(outfile)) and not args.force:
            logging.info("Found feedme file; skipping...")
            continue

        logfile = os.path.join(outdir,'feedme.log')
        cmd = f'feedme.py {args.config} -o {objid}'
        if args.force: cmd += ' -f'
        if args.verbose: cmd += ' -v'

        sub = f'csub -q {args.queue} -o {logfile} -n {args.njobs} {cmd}'
        logging.debug(sub)
        subprocess.call(sub,shell=True)
