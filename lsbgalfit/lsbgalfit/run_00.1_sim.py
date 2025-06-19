#!/usr/bin/env python
"""
Pipeline script for creating image cutouts.
"""
__author__ = "Alex Drlica-Wagner"
import os.path
import logging
import subprocess

import yaml
import numpy as np

import astropy.io.fits as pyfits

RUN = ['all','cutout','feedme','galfit']

if __name__ == "__main__":
    import parser
    parser = parser.Parser(description=__doc__)
    parser.add_argument('-r','--run',action='append',type=str,
                        choices=RUN, help='analysis section to run')
    args = parser.parse_args()

    if not args.run:      args.run = ['all']
    if 'all' in args.run: args.run = RUN[1:]
    logging.info(f"Running analysis steps: {args.run}")

    force = '-f' if args.force else ''
    verbose = '-v' if args.verbose else ''
    
    config = yaml.safe_load(open(args.config))

    cat = pyfits.open(config['catfile'])[1].data.view(np.recarray)
    objids = cat['COADD_OBJECT_ID']

    # Temporarily only select customs
    #logging.info("WARNING: Loading custom constraints...")
    #custom=yaml.safe_load(open(os.path.join(config['tempdir'],config['custfile'])))
    #objids = objids[np.in1d(objids,list(custom.keys()))]

    if args.objid:
        if np.any(~np.in1d(args.objid,objids)):
            raise Exception("COADD_OBJECT_ID not found")
        objids = np.asarray(args.objid).flatten()

    objids = objids[slice(args.imin,args.imax)]
    logging.info("Running %i objects..."%len(objids))
    for i,objid in enumerate(objids):
        resdir = config['galdir'].format(objid=objid)
        resfile = os.path.join(resdir,config['resfile']).format(objid=objid)
        if os.path.exists(resfile) and not args.force:
            logging.info(f"Found {resfile}; skipping...")
            continue

        outdir = config['cutdir'].format(objid=objid)
        if not os.path.exists(outdir): os.makedirs(outdir)
        logfile = os.path.join(outdir,config['logfile']).format(objid=objid)

        cmd  = '"'
        if 'cutout' in args.run:
            cmd += f'cutout.py {args.config} {force} {verbose} -o {objid}; '
        if 'feedme' in args.run:
            cmd += f'feedme.py {args.config} {force} {verbose} -o {objid}; '
        if 'galfit' in args.run:
            cmd += f'galfitm.py {args.config} {force} {verbose} -o {objid}; '
        cmd += '"'

        sub  = f'csub -q {args.queue} -o {logfile} -n {args.njobs} {cmd}'
        logging.debug(sub)
        subprocess.call(sub,shell=True)

