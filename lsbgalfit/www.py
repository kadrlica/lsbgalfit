#!/usr/bin/env python
"""
Create html webpage
"""
__author__ = "Alex Drlica-Wagner"
import os
import glob
import logging

import yaml
import numpy as np
import pandas as pd

import astropy.io.fits as pyfits

INDEX = """
<html>
  <head>
    <title>DES Y6 LSB Galaxies</title>
</head>
<body>
%(table)s
</body>
</html>
"""

TABLE = """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>Fit Info</th>
      <th>GalFit Diagnostics</th>
      <th>SExtractor Segmap</th>
    </tr>
  </thead>
  <tbody>
%(rows)s
  </tbody>
"""

ROW = """
    <tr>
      <th>{idx}</th>
      <td>
        <h4>COADD_OBJECT_ID: <a href="http://legacysurvey.org/viewer/?ra={RA}&dec={DEC}&layer=ls-dr10&zoom=15">{COADD_OBJECT_ID}</a></h4>
        <h4>RA,DEC: {RA:.4f}, {DEC:.4f}</h4>
        <h4>MAG_AUTO (g,r,i): {MAG_AUTO_G:.2f}, {MAG_AUTO_R:.2f}, {MAG_AUTO_I:.2f}</h4>
        <h4>MAG_GALFIT (g,r,i): {MAG_G:.2f}, {MAG_R:.2f}, {MAG_I:.2f}</h4>
        <h4>COLOR_AUTO (g-r,g-i): {COLOR_AUTO_GR:.2f}, {COLOR_AUTO_GI:.2f}</h4>
        <h4>COLOR_GALFIT (g-r,g-i): {COLOR_GR:.2f}, {COLOR_GI:.2f}</h4>
        <h4>FLUX_RADIUS (g,r,i): {FLUX_RADIUS_G:.2f}, {FLUX_RADIUS_R:.2f}, {FLUX_RADIUS_I:.2f}</h4>
        <h4>RE_G,RE_R,RE_I: {RE_G:.2f}, {RE_R:.2f}, {RE_I:.2f}</h4>
        <h4>N_GALFIT: {N:.2f}</h4>
        <h4>CHI2NU: {CHI2NU:.2f}</h4>
        <h4>CHI2NU (g,r,i): {CHI2NU_G:.2f},{CHI2NU_R:.2f},{CHI2NU_I:.2f}</h4>
      </td>
      <td> <a id="{COADD_OBJECT_ID}"></a><a href="{results}"><img src="{results}" alt="LSB Galaxy Fit" width="600"></a></td>
      <td> <a id="{COADD_OBJECT_ID}"></a><a href="{segmap}"><img src="{segmap}" alt="LSB Galaxy Segmap" width="500"></a></td>
    </tr>  
"""

def check_formula(formula):
    """ Check that a formula is valid. """
    if not (('cat[' in formula) and (']' in formula) or ('star_lsb' in formula)):
        msg = f'Invalid formula:\n {formula}'
        raise ValueError(msg)

def eval_formula(cat,formula):
    logging.info(f"  Evaluating selection: {formula}")
    check_formula(formula)
    if formula == 'star_lsb':
        sel = pyfits.open(config['basedir']+'/'+config['starfile'])[1].data.view(np.recarray)
        return sel
    sel = eval(formula)
    return sel

if __name__ == "__main__":
    import parser
    parser = parser.Parser(description=__doc__)
    parser.add_argument("-s","--select",default=None)
    parser.add_argument('--outfile',default=None)
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config))
    cat = pyfits.open(config['basedir']+'/'+config['outfile'])[1].data.view(np.recarray)
    
    if args.select:
        sel = eval_formula(cat,args.select)
        if args.select == 'star_lsb':
            cat = sel
        else:
            cat = cat[sel]

    objids = cat['COADD_OBJECT_ID']
    if args.objid:
        if np.any(~np.in1d(args.objid,objids)):
            raise Exception("COADD_OBJECT_ID not found")
        objids = np.array(args.objid)

    objects = cat[np.in1d(cat['COADD_OBJECT_ID'],objids)]
    objects = objects[slice(args.imin,args.imax)]
    logging.info(f"Assembling table from {len(objects)} objects...")

    tablerows = []
    for i,obj in enumerate(objects):
        objid = obj['COADD_OBJECT_ID']

        results = config['resfile'].format(objid=objid).replace('.fits','.png')
        segmap = results.replace('results','segmap')
        params = {k:obj[k] for k in obj.dtype.names}
        params['results'] = results
        params['segmap'] = segmap
        params['idx'] = i
        params['COLOR_AUTO_GR'] = obj['MAG_AUTO_G']-obj['MAG_AUTO_R']
        params['COLOR_AUTO_GI'] = obj['MAG_AUTO_G']-obj['MAG_AUTO_I']
        params['COLOR_GR'] = obj['MAG_G']-obj['MAG_R']
        params['COLOR_GI'] = obj['MAG_G']-obj['MAG_I']

        tablerows.append(ROW.format(**params))

    table = TABLE%dict(rows='\n'.join(tablerows))
    index = INDEX%dict(table=table)

    logging.info(f"Writing {args.outfile}...")
    with open(args.outfile,'w') as out:
        out.write(index)

    df = pd.DataFrame({'COADD_OBJECT_ID':objects['COADD_OBJECT_ID']})
    outfile = args.outfile.replace('.html','.csv')
    df.to_csv(outfile,index=False)
