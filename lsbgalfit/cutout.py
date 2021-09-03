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

from scipy import ndimage

from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
warnings.simplefilter('ignore', UserWarning)

import galsim
from galsim.des.des_psfex import DES_PSFEx

BANDS = ['g','r','i','z','Y','det']
KEYWORDS = ['TILENAME','BAND','FILTER','MAGZERO','DESFNAME','UNITNAME','ATTNUM','REQNUM']

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

    """
    objid = obj['COADD_OBJECT_ID']
    segmap = copy.deepcopy(segmap)

    logging.info("Clearing object from segmentation map...")
    # First, clear the segmap for this object
    segmap[segmap == obj['OBJECT_NUMBER']] = 0

    if custom is not None:
        try: 
            group = cat[np.in1d(cat['COADD_OBJECT_ID'],custom[objid]['group'])]
            logging.info(f"Clearing custom group from segmap.")
            idx = np.in1d(segmap,group['OBJECT_NUMBER']).reshape(segmap.shape)
            segmap[idx] = 0
        except KeyError: pass

    if True:
        return segmap

    # Then try to figure out which siblings to clear
    group = cat[cat['LSBG_COADD_OBJECT_ID'] == obj['COADD_OBJECT_ID']]
    ngroup = len(group)

    # Only deal with large objects
    flux_radius = np.max([obj[f'FLUX_RADIUS_{band.upper()}'] for band in bands])
    if flux_radius < 15:
        logging.debug("Skipping group...")
        logging.debug("  flux_radius={flux_radius:.2f}")
        return segmap
                      
    # Only deal with objects with lots of siblings
    if ngroup < 4: 
        logging.debug("Skipping group...")
        logging.debug("  Ngroup = {ngroup}")
        return segmap

    for o in group:
        # Ignore likely stars
        if np.abs(o['EXTENDED_CLASS_MASH_SOF']) < 3: 
            logging.debug(f"  extended_class_mash_sof={o['EXTENDED_CLASS_MASH_SOF']}")
            continue
        
        # Ignore small objects
        if np.sqrt(np.abs(o['SOF_CM_T'])) < 5: 
            logging.debug(f"  sof_cm_t={o['SOF_CM_T']:.2f}")
            continue

        # Separation too large
        sep2=(o['XWIN_IMAGE']-obj['XWIN_IMAGE'])**2+(o['YWIN_IMAGE']-obj['YWIN_IMAGE'])**2
        if np.sqrt(sep2) > 2.5*flux_radius: 
            logging.debug(f"  sep={np.sqrt(sep2):.2f}, flux_radius={flux_radius:.2f}")
            continue

        logging.info("Clearing sibling from segmap: {}".format(o['COADD_OBJECT_ID']))
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
    out = np.in1d(label,uid[cts > mincts]).reshape(label.shape) > 0
    out = ndimage.binary_dilation(out)
    
    return out

def extra_mask(wcs,mask,obj,custom):
    objid = obj['COADD_OBJECT_ID']
    extra = np.zeros_like(mask)
    try: tomask = custom[objid]['mask']
    except KeyError: return extra

    # Not 100% sure on the ordering here...
    xx,yy = np.meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
    ra,dec = wcs.all_pix2world(xx,yy,0)

    for x in tomask:
        if x['type'] != 'circle':
            logging.warn("Unrecognized mask type: '%s'"%x['type'])
            continue
        logging.info(f"Masking object at: {x['x']}, {x['y']}")
        sel = (angsep(x['x'],x['y'],ra,dec) < x['radius'])

        #import pdb; pdb.set_trace()
        
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
    plt.savefig(outfile,dpi=150)

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

    group = pyfits.open(config['grpfile'])[1].data.view(np.recarray)
    cat = pyfits.open(config['catfile'])[1].data.view(np.recarray)
    objids = np.asarray(args.objid)

    custom = yaml.safe_load(open(os.path.join(config['tempdir'],config['custfile'])))
                
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

            imgfile = glob.glob(tilepath+f'/*_{band}.fits.fz')[0]
            psffile = glob.glob(tilepath+f'/*_{band}_psfcat.psf')[0]

            logging.info("Reading %s..."%imgfile)
            img = pyfits.open(imgfile)

            # Only read when necessary to save memory
            #img = img["SCI"].data
            #msk = img["MSK"].data
            #wgt = img["WGT"].data

            wcs = WCS(pyfits.getheader(imgfile, ext=1))
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
            
            # Mask
            logging.debug("Creating bad mask...")

            mask = (segcut.data > 0) | (mskcut.data > 0)

            # Fill any holes
            holes = fill_holes(mask)
            mask[holes > 0] = True

            # Find any missing sources mostly g-band
            sources = find_sources(imgcut.data,mask)

            # Clear the segmap for the source/group of interest
            bad = clear_segmap(segcut.data,obj,group,bands,custom=custom)

            # Any extra custom masking
            extra = extra_mask(mskcut.wcs,mask,obj,custom)

            # Combine everything...
            bad[mskcut.data > 0] = 1
            bad[sources > 0] = 1
            bad[holes > 0] = 1
            bad[bad > 0] = 1
            bad[extra > 0 ] = 1

            logging.debug("Creating sigma image...")
            # The DES weight plane is an inverse variance image
            sigcut = 1/np.sqrt(copy.deepcopy(wgtcut.data))

            # Write the psf image
            logging.debug("Creating psf image...")
            pos = galsim.PositionD(x,y)
            psfimg = psf.getPSFArray(pos)

            #hdu = pyfits.PrimaryHDU(imgcut.data)
            #hdu.header.update(hdr)
            #hdu.header.update(imgcut.wcs.to_header())         
            #hdu.header.update(EXTNAME='SCI')
            #f = os.path.join(outdir,config['imgfile'].format(objid=objid,band=band))
            #logging.info("Writing %s..."%f)
            #hdu.writeto(f, overwrite=True)
            # 
            #hdu = pyfits.PrimaryHDU(bad)
            #hdu.header.update(copy.deepcopy(img['MSK'].header))
            #hdu.header.update(mskcut.wcs.to_header())
            #hdu.header.update(EXTNAME="BAD")
            #f = os.path.join(outdir,config['mskfile'].format(objid=objid,band=band))
            #logging.info("Writing %s..."%f)
            #hdu.writeto(f, overwrite=True)

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

