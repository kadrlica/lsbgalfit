#!/usr/bin/env python
"""
Create cutouts
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
from astropy.wcs import WCS
#from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
warnings.simplefilter('ignore', UserWarning)
#from astropy.utils.compat.optional_deps import HAS_BOTTLENECK

import galsim
from galsim.des.des_psfex import DES_PSFEx

BANDS = ['g','r','i','z','Y','det']
KEYWORDS = ['TILENAME','BAND','FILTER','MAGZERO','DESFNAME','UNITNAME','ATTNUM','REQNUM']

# Monkeypatch for sigma_clipped_stats
def sigma_clipped_stats(
    data,
    mask=None,
    mask_value=None,
    sigma=3.0,
    sigma_lower=None,
    sigma_upper=None,
    maxiters=5,
    cenfunc="median",
    stdfunc="std",
    std_ddof=0,
    axis=None,
    grow=False,
):
    """
    Calculate sigma-clipped statistics on the provided data.

    Parameters
    ----------
    data : array-like or `~numpy.ma.MaskedArray`
        Data array or object that can be converted to an array.

    mask : `numpy.ndarray` (bool), optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are excluded when computing the statistics.

    mask_value : float, optional
        A data value (e.g., ``0.0``) that is ignored when computing the
        statistics. ``mask_value`` will be masked in addition to any
        input ``mask``.

    sigma : float, optional
        The number of standard deviations to use for both the lower
        and upper clipping limit. These limits are overridden by
        ``sigma_lower`` and ``sigma_upper``, if input. The default is 3.

    sigma_lower : float or None, optional
        The number of standard deviations to use as the lower bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    sigma_upper : float or None, optional
        The number of standard deviations to use as the upper bound for
        the clipping limit. If `None` then the value of ``sigma`` is
        used. The default is `None`.

    maxiters : int or None, optional
        The maximum number of sigma-clipping iterations to perform or
        `None` to clip until convergence is achieved (i.e., iterate
        until the last iteration clips nothing). If convergence is
        achieved prior to ``maxiters`` iterations, the clipping
        iterations will stop. The default is 5.

    cenfunc : {'median', 'mean'} or callable, optional
        The statistic or callable function/object used to compute
        the center value for the clipping. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanmean`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'median'``.

    stdfunc : {'std', 'mad_std'} or callable, optional
        The statistic or callable function/object used to compute the
        standard deviation about the center value. If using a callable
        function/object and the ``axis`` keyword is used, then it must
        be able to ignore NaNs (e.g., `numpy.nanstd`) and it must have
        an ``axis`` keyword to return an array with axis dimension(s)
        removed. The default is ``'std'``.

    std_ddof : int, optional
        The delta degrees of freedom for the standard deviation
        calculation. The divisor used in the calculation is ``N -
        std_ddof``, where ``N`` represents the number of elements. The
        default is 0.

    axis : None or int or tuple of int, optional
        The axis or axes along which to sigma clip the data. If `None`,
        then the flattened data will be used. ``axis`` is passed to the
        ``cenfunc`` and ``stdfunc``. The default is `None`.

    grow : float or `False`, optional
        Radius within which to mask the neighbouring pixels of those
        that fall outwith the clipping limits (only applied along
        ``axis``, if specified). As an example, for a 2D image a value
        of 1 will mask the nearest pixels in a cross pattern around each
        deviant pixel, while 1.5 will also reject the nearest diagonal
        neighbours and so on.

    Notes
    -----
    The best performance will typically be obtained by setting
    ``cenfunc`` and ``stdfunc`` to one of the built-in functions
    specified as as string. If one of the options is set to a string
    while the other has a custom callable, you may in some cases see
    better performance if you have the `bottleneck`_ package installed.

    .. _bottleneck:  https://github.com/pydata/bottleneck

    Returns
    -------
    mean, median, stddev : float
        The mean, median, and standard deviation of the sigma-clipped
        data.

    See Also
    --------
    SigmaClip, sigma_clip
    """
    if mask is not None:
        data = np.ma.MaskedArray(data, mask)
    if mask_value is not None:
        data = np.ma.masked_values(data, mask_value)

    if isinstance(data, np.ma.MaskedArray) and data.mask.all():
        return np.ma.masked, np.ma.masked, np.ma.masked

    sigclip = SigmaClip(
        sigma=sigma,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        cenfunc=cenfunc,
        stdfunc=stdfunc,
    )
    data_clipped = sigclip(
        data, axis=axis, masked=False, return_bounds=False, copy=True
    )

    #if HAS_BOTTLENECK:
        #mean = _nanmean(data_clipped, axis=axis)
       # median = _nanmedian(data_clipped, axis=axis)
        #std = _nanstd(data_clipped, ddof=std_ddof, axis=axis)
    if True:  # pragma: no cover
        mean = np.nanmean(data_clipped, axis=axis)
        median = np.nanmedian(data_clipped, axis=axis)
        std = np.nanstd(data_clipped, ddof=std_ddof, axis=axis)

    return mean, median, std
import astropy.stats

astropy.stats.sigma_clipped_stats = sigma_clipped_stats
from astropy.stats import sigma_clipped_stats

def angsep(lon1,lat1,lon2,lat2):
    """
    Angular separation (deg) between two sky coordinates.
    Borrowed from astropy (www.astropy.org)

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1],
    which is slighly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    [1] http://en.wikipedia.org/wiki/Great-circle_distance
    """
    lon1,lat1 = np.radians([lon1,lat1])
    lon2,lat2 = np.radians([lon2,lat2])
    
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.degrees(np.arctan2(np.hypot(num1,num2), denominator))

def fill_holes(mask):
    """Fill in holes in the segmap that arise from unmasked saturated
    pixels in bright stars.
    
    See COADD_OBJECT_ID: 78866609, 79093465, 78140711

    Parameters
    ----------
    mask : the mask array (should be integer type)

    Returns
    -------
    filled : the mask array with holes filled
    """
    logging.info("Filling binary holes in bad mask...")
    filled = ndimage.binary_fill_holes(mask).astype(mask.dtype)
    return filled &~ mask

def clear_segmap(segmap,obj,cat,bands,custom=None):
    """Remove objects from the segmentation map. The basic idea of this
    function is to stop shredding large objects. However, things become
    more complicated when we want to be sure not to include the light from
    foreground stars or globular clusters.
    
    From Kai:
    Added functionality to add objects back onto the segmap

    """
    objid = obj['COADD_OBJECT_ID']
    segmap = copy.deepcopy(segmap)

    logging.info("Clearing object from segmentation map...")
    # First, clear the segmap for this object
    segmap[segmap == obj['OBJECT_NUMBER']] = 0

    if custom is not None:
        try:
            group = cat[np.in1d(cat['COADD_OBJECT_ID'],custom[objid]['remove'][0]['group'])]
            logging.info(f"Clearing custom group from segmap.")
            idx = np.in1d(segmap,group['OBJECT_NUMBER']).reshape(segmap.shape)
            segmap[idx] = 0
        except KeyError: pass 

    #if True:
        #return segmap

    # Then try to figure out which siblings to clear
    group = cat[cat['LSBG_COADD_OBJECT_ID'] == obj['COADD_OBJECT_ID']]
    ngroup = len(group)
    logging.debug(f'Found {ngroup} objects in group...')
    
    

    # Only deal with large objects
    flux_radius = np.max([obj[f'FLUX_RADIUS_{band.upper()}'] for band in bands])
    if flux_radius < 15:
        logging.debug("Skipping group...")
        logging.debug(f"flux_radius={flux_radius:.2f}")
        # Re-mask objects that need to be added
        try:
            group = cat[np.in1d(cat['COADD_OBJECT_ID'],custom[objid]['add'][0]['group'])]
            logging.info("Adding custom group from segmap: {}".format(group['COADD_OBJECT_ID']))
            idx = np.in1d(segmap,group['OBJECT_NUMBER']).reshape(segmap.shape)
            segmap[idx] = 1
        except KeyError: pass
        return segmap
                      
    # Only deal with objects with lots of siblings
    if ngroup < 3: 
        logging.debug("Skipping group...")
        logging.debug(f"  Ngroup = {ngroup}")
        # Re-mask objects that need to be added
        try:
            group = cat[np.in1d(cat['COADD_OBJECT_ID'],custom[objid]['add'][0]['group'])]
            logging.info("Adding custom group from segmap: {}".format(group['COADD_OBJECT_ID']))
            idx = np.in1d(segmap,group['OBJECT_NUMBER']).reshape(segmap.shape)
            segmap[idx] = 1
        except KeyError: pass
        return segmap

    for o in group:
        # Ignore likely stars
        if np.abs(o['EXT_MASH']) < 3: 
            logging.debug(f"  extended_class_mash_sof={o['EXT_MASH']}")
            continue
        
        # Ignore small objects
        if np.sqrt(np.abs(o['BDF_T'])) < 5: 
            logging.debug(f"  bdf_t={o['BDF_T']:.2f}")
            continue

        # Separation too large
        sep2=(o['XWIN_IMAGE']-obj['XWIN_IMAGE'])**2+(o['YWIN_IMAGE']-obj['YWIN_IMAGE'])**2
        if np.sqrt(sep2) > 2.5*flux_radius: 
            logging.debug(f"  sep={np.sqrt(sep2):.2f}, flux_radius={flux_radius:.2f}")
            continue

        logging.info("Clearing sibling from segmap: {}".format(o['COADD_OBJECT_ID']))
        # Re-mask objects that need to be added
        try:
            group = cat[np.in1d(cat['COADD_OBJECT_ID'],custom[objid]['add'][0]['group'])]
            if o['COADD_OBJECT_ID'] in group['COADD_OBJECT_ID']:
                logging.info("Adding custom group from segmap: {}".format(group['COADD_OBJECT_ID']))
                idx = np.in1d(segmap,group['OBJECT_NUMBER']).reshape(segmap.shape)
                segmap[idx] = 1
                continue
        except KeyError: pass
        segmap[segmap == o['OBJECT_NUMBER']] = 0

    return segmap

def find_sources(data, mask, nsigma = 10, mincts = 25):
    """Find sufficiently bright sources and mask them"""
    logging.info("Finding bright sources to mask...")

    mean, median, stddev = sigma_clipped_stats(data, mask=(mask>0), sigma=3)

    search = copy.deepcopy(data)
    search[mask>0] = median
    search -= median

    label, nlabel = ndimage.label(search > nsigma*stddev)
    uid, cts = np.unique(label, return_counts=True)
    uid,cts = uid[1:],cts[1:]

    logging.info("Masking %s bright sources"%( (cts>mincts).sum() ))
    out = np.in1d(label,uid[cts > mincts]).reshape(label.shape) > 0
    out = ndimage.binary_dilation(out)
    
    return out

def extra_mask(wcs,mask,obj,custom):
    objid = obj['COADD_OBJECT_ID']
    extra = np.zeros_like(mask)

    if not custom: return extra
    try: tomask = custom[objid]['mask']
    except KeyError: return extra

    # Not 100% sure on the ordering here...
    xx,yy = np.meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
    ra,dec = wcs.all_pix2world(xx,yy,0)

    for x in tomask:
        if x['type'] != 'circle':
            logging.warn("Unrecognized mask type: '%s'"%x['type'])
            continue
        if x['unit'] == 'pix':
            logging.info('Setting unit to pixel...')
            sky = wcs.pixel_to_world(x['x'],x['y'])
            ra_x = sky.ra.degree
            dec_y = sky.dec.degree
            sky = wcs.pixel_to_world(x['x']+x['radius'],x['y'])
            radius = abs(sky.ra.degree - ra_x)
        elif x['unit'] == 'deg':
            logging.info('Setting unit to degree...')
            ra_x = x['x']
            dec_y = x['y']
            radius = x['radius']
        else:
            unit = x['unit']
            logging.warn(f'Unrecognized coordinate type: {unit}')
            continue
        logging.info(f"Masking object at: {ra_x}, {dec_y}")
        #sel = (angsep(x['x'],x['y'],ra,dec) < x['radius'])
        sel = (angsep(ra_x,dec_y,ra,dec) < radius)
        logging.info(f"Circle drawn with radius: {x['radius']}")

        extra[sel] = True

    return extra

def plot_segmap(config,obj,grp):
    objid = obj['COADD_OBJECT_ID']
    group = grp[grp['LSBG_COADD_OBJECT_ID']==objid]

    cutfile = os.path.join(config['cutdir'],config['cutfile']).format(objid=objid,band='g')
    f = pyfits.open(cutfile)
    img = f['SCI'].data
    seg = f['SEG'].data
    header = f['SEG'].header
    wcs = WCS(header)

    size = min(5*obj['FLUX_RADIUS_G'], img.shape[0]-1)

    x,y = wcs.wcs_world2pix(obj['RA'],obj['DEC'],0)
    imgcut = Cutout2D(img, position=(x, y), size=(size,size),wcs=wcs,
                      mode='partial',fill_value=np.nan, copy=True)
    segcut = Cutout2D(seg, position=(x, y), size=(size,size),wcs=wcs,
                      mode='partial',fill_value=np.nan, copy=True)

    fig = plt.figure(figsize=(8,6))

    norm = viz.ImageNormalize(imgcut.data, interval=viz.ZScaleInterval())
    plt.imshow(imgcut.data,origin='lower',norm=norm,cmap='gray')
    print(imgcut.data.shape)

    vmin = group['OBJECT_NUMBER'].min() - 10
    vmax = group['OBJECT_NUMBER'].max() + 10

    array = np.ma.array(segcut.data,mask=segcut.data==0)
    print(segcut.data.shape)
    print(array.shape)
    plt.imshow(array,origin='lower',cmap='tab20',vmin=vmin,vmax=vmax,alpha=0.3)
    plt.colorbar()

    color='chartreuse'
    x,y = segcut.wcs.wcs_world2pix(obj['RA'],obj['DEC'],0)
    plt.annotate(obj['COADD_OBJECT_ID'],(x,y),fontsize=12,weight='bold',color=color)
    plt.scatter(x,y,marker='o',s=5,c=color,facecolor='none',edgecolor=color)
    
    for o in group:
        if o['COADD_OBJECT_ID'] == objid: continue
        x,y = segcut.wcs.wcs_world2pix(o['RA'],o['DEC'],0)
        plt.annotate(o['COADD_OBJECT_ID'],(x,y),fontsize=8,color=color)
        plt.scatter(x,y,marker='x',s=3,c=color)

    plt.xlim(0,size)
    plt.ylim(0,size)

    plt.suptitle(objid,y=0.98)

    outfile = os.path.join(config['galdir'],'segmap_{objid}.png').format(objid=objid)
    logging.info("Writing %s..."%outfile)
    plt.savefig(outfile,dpi=100)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('-o','--objid',nargs='+',
                        type=int,required=True,
                        help='cutouts by coadd_object_id')
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
    objids = np.asarray(args.objid)


    if config['grpfile']:
        group = pyfits.open(config['grpfile'])[1].data.view(np.recarray)
    else:
        logging.debug('No groupfile found...')
        group = None
    
    if config['custfile']:
        custom = yaml.safe_load(open(os.path.join(config['tempdir'],config['custfile'])))
    else:
        custom = None
                
    print("Generating cutouts for %i objects..."%len(objids))
    objects = cat[np.in1d(cat['COADD_OBJECT_ID'],objids)]

    for obj in objects:
        objid = obj['COADD_OBJECT_ID']
        tilename = obj['TILENAME'].astype(str)
        tilepath = os.path.join(config['datadir'],tilename)
        logging.info(f"{tilename}: {objid}")
        
        outdir = config['cutdir'].format(objid=objid)
        if not os.path.exists(outdir): os.makedirs(outdir)

        # Set the cutout size based on the minimum size in the config
        # and the flux radius (retaining even/odd size)
        flux_radius = np.max([obj[f'FLUX_RADIUS_{band.upper()}'] for band in bands])
        
        # Set object size if there is a custom size being given
        try:
            size = custom[obj['COADD_OBJECT_ID']]['size']
            logging.info(f'Custom size given: {size}')
        except KeyError:
            size = config['size']
            size = max(size, (flux_radius//5 + 1) * 50 + size%2)

        x,y = obj.XWIN_IMAGE,obj.YWIN_IMAGE
        logging.debug(f"Object located at: ({x:.2f}, {y:.2f})")
        logging.debug(f"Cutout size: {size}")

        segfile = glob.glob(tilepath+'/*_det_segmap.fits')[0]
        logging.info("Reading %s..."%segfile)
        seg  = pyfits.open(segfile)

        for i,band in enumerate(bands):
            outbase = config['cutfile'].format(objid=objid,band=band)
            outfile = os.path.join(outdir,outbase)
            if os.path.exists(outfile) and not args.force:
                logging.info("Found %s; skipping..."%(outfile))
                continue

            # Pick up fits or fits.fz
            imgfile = glob.glob(tilepath+f'/*_{band}.fits.fz')[0]
            psffile = glob.glob(tilepath+f'/*_{band}_psfcat.psf')[0]

            logging.info("Reading %s..."%imgfile)
            img = pyfits.open(imgfile)

            # Only read when necessary to save memory
            #img = img["SCI"].data
            #msk = img["MSK"].data
            #wgt = img["WGT"].data

            wcs = WCS(pyfits.getheader(imgfile, extname='SCI'))
            psf = DES_PSFEx(psffile)
            hdr = {k:img['SCI'].header[k] for k in KEYWORDS}

            logging.debug("Creating cutout...")
            imgcut = Cutout2D(img["SCI"].data, position=(x, y), size=(size,size), 
                              wcs=wcs,mode='partial',fill_value=np.nan, copy=True)
                              
            mskcut = Cutout2D(img["MSK"].data, position=(x, y), size=(size,size), 
                              wcs=wcs, mode='partial',fill_value=1, copy=True)
                              
            wgtcut = Cutout2D(img["WGT"].data, position=(x, y), size=(size,size), 
                              wcs=wcs, mode='partial',fill_value=np.nan, copy=True)
                              
            segcut = Cutout2D(seg["SCI"].data, position=(x, y), size=(size,size), 
                              wcs=wcs, mode='partial',fill_value=1000000, copy=True)

            simdir = config.get('simdir')
            if simdir:
                logging.info("Replacing image with simulated data...")
                simfile = glob.glob(simdir+f'/*_added_{band}.fits')[0]
                #simfile = glob.glob(simdir+f'/*_temp_{band}.fits')[0]
                sim = pyfits.open(simfile)
                imgcut = Cutout2D(sim["SCI"].data, position=(x, y), size=(size,size),
                                  wcs=wcs,mode='partial',fill_value=np.nan, copy=True)

            # Mask
            logging.debug("Creating bad mask...")

            mask = (segcut.data > 0) | (mskcut.data > 0)

            # Fill any holes
            holes = fill_holes(mask)
            mask[holes > 0] = True

            # Find any missing sources mostly g-band
            sources = find_sources(imgcut.data,mask,nsigma=np.inf)

            # Clear the segmap for the source/group of interest
            bad = clear_segmap(segcut.data,obj,group,bands,custom=custom)
            

            # Any extra custom masking
            extra = extra_mask(mskcut.wcs,mask,obj,custom)

            # Combine everything...
            bad[mskcut.data > 0] = 1
            bad[sources > 0] = 1
            bad[holes > 0] = 1
            bad[bad > 0] = 1
            p = len(bad[bad>0])
            logging.debug(f'Before extra:{p}')
            bad[extra > 0] = 1
            p = len(bad[bad>0])
            logging.debug(f'After extra:{p}')

            logging.debug("Creating sigma image...")
            # The DES weight plane is an inverse variance image
            with np.errstate(divide='ignore'):
                sigcut = 1/np.sqrt(copy.deepcopy(wgtcut.data))
            #sigcut = 1e5 * 1/np.sqrt(copy.deepcopy(wgtcut.data))

            # Write the psf image
            logging.debug("Creating psf image...")
            pos = galsim.PositionD(x,y)
            psfimg = psf.getPSFArray(pos)

            # Write multi-extension output
            hdulist = []

            hdu = pyfits.PrimaryHDU(imgcut.data)
            hdu.header.update(hdr)
            hdu.header.update(imgcut.wcs.to_header())
            hdu.header.update(EXTNAME='SCI',OBJID=objid)
            hdulist.append(hdu)

            hdu = pyfits.ImageHDU(mskcut.data)
            hdu.header.update(hdr)
            hdu.header.update(mskcut.wcs.to_header())
            hdu.header.update(EXTNAME='MSK',OBJID=objid)
            hdulist.append(hdu)

            hdu = pyfits.ImageHDU(wgtcut.data)
            hdu.header.update(hdr)
            hdu.header.update(wgtcut.wcs.to_header())
            hdu.header.update(EXTNAME='WGT',OBJID=objid)
            hdulist.append(hdu)

            hdu = pyfits.ImageHDU(segcut.data)
            hdu.header.update(hdr)
            hdu.header.update(segcut.wcs.to_header())
            hdu.header.update(EXTNAME='SEG',OBJID=objid)
            hdulist.append(hdu)

            hdu = pyfits.ImageHDU(bad)
            hdu.header.update(hdr)
            hdu.header.update(mskcut.wcs.to_header())
            hdu.header.update(EXTNAME='BAD',OBJID=objid)
            hdulist.append(hdu)

            hdu = pyfits.ImageHDU(sigcut)
            hdu.header.update(hdr)
            hdu.header.update(wgtcut.wcs.to_header())
            hdu.header.update(EXTNAME='SIG',OBJID=objid)
            hdulist.append(hdu)

            hdu = pyfits.ImageHDU(psfimg)
            hdu.header.update(EXTNAME='PSF',OBJID=objid)
            hdulist.append(hdu)

            logging.info("Writing %s..."%outfile)
            pyfits.HDUList(hdulist).writeto(outfile, overwrite=True)
             
            #hdu = pyfits.PrimaryHDU(psfimg)
            #outbase = config['psffile'].format(objid=objid,band=band)
            #outfile = os.path.join(outdir,outbase)
            #logging.info("Writing %s..."%outfile)
            #hdu.writeto(outfile, overwrite=True)

