#!/usr/bin/env python
"""
Pipeline script for running galfit.
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
        if not np.any(np.in1d(args.objid,objids)):
            raise Exception("COADD_OBJECT_ID not found")
        objids = np.array(args.objid)

    objids = objids[slice(args.imin,args.imax)]
    for i,objid in enumerate(objids):
        outdir = config['galdir'].format(objid=objid)
        galfile = os.path.join(outdir,config['galfile']).format(objid=objid)
        resfile = os.path.join(outdir,config['resfile']).format(objid=objid)
        feedme = os.path.join(outdir,config['feedme'])

        if os.path.exists(resfile) and not args.force:
            logging.info("Found %s; skipping..."%resfile)
            continue

        if not os.path.exists(outdir): 
            logging.warn("Missing %s; skipping"%outdir)
            continue

        if not os.path.exists(feedme):
            logging.warn("Missing %s; skipping"%feedme)
            continue

        logfile = os.path.join(outdir,'galfitm.log')
        cmd = f'galfitm.py {args.config} -o {objid}'

        sub = f'csub -q {args.queue} -o {logfile} -n {args.njobs} {cmd}'
        logging.debug(sub)
        subprocess.call(sub,shell=True)
