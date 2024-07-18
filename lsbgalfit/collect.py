#!/usr/bin/env python
"""
Collect results
"""
__author__ = "Alex Drlica-Wagner"
import os.path
import glob
import copy
import logging
import warnings
import multiprocessing
from collections import OrderedDict as odict

import yaml
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as recfn

from astropy.io import fits as pyfits
from astropy.wcs import WCS

warnings.simplefilter('ignore', UserWarning)

BANDS = ['G','R','I']

NAMES  = ['XC','YC','N','N_ERR','AR','AR_ERR','PA','PA_ERR']
TYPES  = [(name,'>f4') for name in NAMES]

#Per-band columns
BNAMES = ['MAG','MAG_ERR','CHI2NU','RE','RE_ERR']
BTYPES = [(f'{name}_{band}','>f4') for band in BANDS for name in BNAMES]

#Extended column names
FTYPES = [('CHI2NU','>f4'),('NFREE','>i4'),('NFIX','>i4'),('NITER','>i4'),('FLAGS','<U81')]
FNAMES = [name for name,dtype in FTYPES]

#Extended columns
ENAMES = ['MU_MEAN','MU_0']
ETYPES = [('ALPHA_J2000','>f8'),('DELTA_J2000','>f8'),('CUSTOM','>i2')]
ETYPES += [(f'{name}_{band}','>f4') for band in BANDS for name in ENAMES]
ENAMES = [name for name,dtype in ETYPES]

def sersic_mueff(mag, reff):
    """
    Mean effective surface brightness. 

    Careful: you decide if reff is circularized or not...

    From Graham's notes:
    https://ned.ipac.caltech.edu/level5/March05/Graham/Graham2_2.html
    
    Parameters
    ----------
    mag  : magnitude (measured from total flux
    reff : half-light (half-flux) radius [arcsec]
    
    Returns
    -------
    mueff : mean effective surface brightness [mag/arcsec^2]
    """
    # Extra factor of 2 to correct for flux outside reff
    return mag + 2.5*np.log10(2 * np.pi * reff**2)

def sersic_mu0(mueff,n):
    """
    Approximate Sersic central surface brightness.

    From Graham's notes:
    https://ned.ipac.caltech.edu/level5/March05/Graham/Graham2_2.html

      mu0 = mueff - 2.5 bn / ln(10)

    Parameters
    ----------
    mueff : mean effective surface brightness [mag/arcsec^2]
    n     : Sersic index
    
    Returns
    -------
    mu0 : approximate central surface brightness [mag/arcsec^2]
    """
    return mueff - 2.5 / np.log(10) * sersic_bn(n)[:,np.newaxis]
    
def sersic_bn(n):
    """
    An approximate value for bn is provided in Equation A1 of
    (really a reference to Ciotti & Bertin 1999)

      bn ~ 2n - 1/3 + 4/405n + 46/25515n**2 + 131/1148175n**3 - 2194697/30690717750n**4

    For n < 0.36 MacArthur, Courteau & Holtzman 2003 provide the following 
    polynomial fit (http://adsabs.harvard.edu/abs/2003ApJ...582..689):

      a0=0.01945; a1=-0.8902; a2=10.95; a3=-19.67; a4=13.43;

    Also see a numerical table from:
    http://www.astr.tohoku.ac.jp/~akhlaghi/Sersic_C.html

    Parameters
    ----------
    n : Sersic index
    
    Returns
    -------
    bn : prefactor
    """
    def big_n(n):
        return 2*n - 1./3 + 4/405. * n**-1 + 46/25515. * n**-2 \
            + 131/1148175 * n**-3 \
            - 2194697/30690717750. * n**-4

    def small_n(n):
        return 0.01945 - 0.8902*n + 10.95*n**2 - 19.67*n**3 + 13.43*n**4

    if np.isscalar(n):
        return np.asscalar(np.where(n > 0.36, big_n(n), small_n(n)))
    else:
        return np.where(n > 0.36, big_n(n), small_n(n))


def read_result(args):
    """Read results from galfit output."""
    config,obj = args
    objid = obj['COADD_OBJECT_ID']
    logging.debug(f"Reading {objid}...")

    bands = config['bands']
    ubands = list(map(str.upper,bands))

    dtype = [('COADD_OBJECT_ID','>i8')]
    dtype += TYPES + BTYPES + FTYPES + ETYPES

    out = np.recarray(1,dtype=dtype)
    out.fill(np.nan)
    out['COADD_OBJECT_ID'] = objid
    out['FLAGS'] = ''

    custom=yaml.safe_load(open(os.path.join(config['tempdir'],config['custfile'])))
    if custom and objid in custom.keys():
        out['CUSTOM'] = True

    resfile = os.path.join(config['galdir'],config['resfile']).format(objid=objid)
    if not os.path.exists(resfile): 
        logging.debug("  Results do not exist; skipping...")
        return out
    try: 
        fits = pyfits.open(resfile)
        final_band = fits['FINAL_BAND'].data
        for name in NAMES:
            out[name] = final_band['COMP1_'+name][0]
        for name in BNAMES:
            for i,band in enumerate(ubands):
                out[f'{name}_{band}'] = final_band['COMP1_'+name][i]
        fit_info = fits['FIT_INFO'].data
        for name in FNAMES:
            out[name] = fit_info[name]

    except:
        logging.debug("Failed to get results; skipping...")

    #Other derived properties
    for band in ubands:
        reff = out[f"RE_{band}"]*np.sqrt(out[f"AR"])*0.263
        mueff=sersic_mueff(out[f"MAG_{band}"], reff)
        mu0 = sersic_mu0(mueff,out["N"])
        out[f"MU_MEAN_{band}"]=mueff
        out[f"MU_0_{band}"] = mu0
     
    wcs = WCS(fits['INPUT_%s'%bands[0]].header)
    ra,dec = wcs.all_pix2world(out['XC'],out['YC'],0)
    out["ALPHA_J2000"] = ra
    out["DELTA_J2000"] = dec
        
    return out


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

    objects = cat[np.in1d(cat['COADD_OBJECT_ID'],objids)]
    objects = objects[slice(args.imin,args.imax)]

    print("Collecting results for %i objects..."%len(objects))

    arglist = list(zip(len(objects)*[config],objects))

    if args.njobs != 1:
        from multiprocessing import Pool
        processes = args.njobs if args.njobs > 0 else None
        p = Pool(processes,maxtasksperchild=1)
        out = p.map(read_result,arglist)
    else:
        out = [read_result(arg) for arg in arglist]

    dtype = out[0].dtype
    for i,d in enumerate(out):
        if d.dtype != dtype: 
            # ADW: Not really safe...
            logging.warn("Casting input data to same type.")
            out[i] = d.astype(dtype)

    logging.info('Concatenating arrays...')
    results = np.concatenate(out)

    if np.all(results['COADD_OBJECT_ID'] != objects['COADD_OBJECT_ID']):
        raise Exception("COADD_OBJECT_IDs do not match.")

    out = recfn.join_by('COADD_OBJECT_ID',objects,results,usemask=False)

    print("Missing results for %i objects."%np.isnan(out['MAG_G']).sum())

    outfile = config['outfile']
    print(f"Writing {outfile}.")
    hdu = pyfits.BinTableHDU(data=out)
    hdu.writeto(outfile,overwrite=True)
