#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import os.path
from os.path import join
import logging
import subprocess
import glob
import itertools
import warnings
import copy

import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm
import pylab as plt

import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.visualization as viz

def run_galfitm(config,obj):
    """Run galfitm for a specific object"""
    objid = obj['COADD_OBJECT_ID']
    outdir = config['galdir'].format(objid=objid)
    feedme = config['feedfile']
    cmd = f'cd {outdir}; galfitm {feedme}'
    logging.info(cmd)
    subprocess.check_call(cmd,shell=True)

def add_bad_mask(config,obj):
    """Add the mask hdus to the results file"""
    objid = obj['COADD_OBJECT_ID']
    
    filename = os.path.join(config['galdir'],config['resfile']).format(objid=objid)
    hdulist = pyfits.open(filename)

    for i,band in enumerate(config['bands']):
        logging.info(f"Adding bad mask for band {band}")
        cutfile = os.path.join(config['cutdir'],config['cutfile']).format(objid=objid,band=band)
        fits = pyfits.open(cutfile)
        hdu = fits['BAD']
        hdu.header.update(EXTNAME=f'MASK_{band}')
        hdulist.insert(3+i,hdu)

    logging.info(f"Writing {filename}")
    hdulist.writeto(filename,overwrite=True)

def add_sigma_image(config,obj):
    """Add the sigma hdus to the results file"""
    objid = obj['COADD_OBJECT_ID']
    
    filename = os.path.join(config['galdir'],config['resfile']).format(objid=objid)
    hdulist = pyfits.open(filename)

    for i,band in enumerate(config['bands']):
        logging.info(f"Adding sigma image for band {band}")
        cutfile = os.path.join(config['cutdir'],config['cutfile']).format(objid=objid,band=band)
        fits = pyfits.open(cutfile)
        hdu = fits['SIG']
        hdu.header.update(EXTNAME=f'SIGMA_{band}')
        hdulist.insert(3+i,hdu)

    logging.info(f"Writing {filename}")
    hdulist.writeto(filename,overwrite=True)

def calc_reduced_chisq(data,model,sigma,mask,nparams):
    """ Calculate the reduced chisq from a set of images.

    Parameters
    ----------
    data : the input data image
    model: the model image
    sigma: the std image
    mask : the mask of pixels to exclude
    nparams: the number of model parameters
    """
    sel = np.where(~mask)
    d = data[sel]
    m = model[sel]
    s = sigma[sel]
    ndof = float(len(d) - nparams)
    chisq = 1/ndof * np.sum( (d - m)**2 / s**2)
    return chisq


def calc_local_chisq(config,obj):
    objid = obj['COADD_OBJECT_ID']
    
    filename = os.path.join(config['galdir'],config['resfile']).format(objid=objid)
    fits = pyfits.open(filename)

    final_band = fits['FINAL_BAND'].data
    local_chisq = -1*np.ones(len(final_band))

    for i,band in enumerate(config['bands']):
        logging.info(f"Calculating local reduced chisq for band {band}")

        x = final_band['COMP1_XC'][i]
        y = final_band['COMP1_YC'][i]
        size = final_band['COMP1_RE'][i]

        data = Cutout2D(fits[f'INPUT_{band}'].data, position=(x, y), size=(size,size), 
                        mode='partial',fill_value=np.nan, copy=True).data
        model = Cutout2D(fits[f'MODEL_{band}'].data, position=(x, y), size=(size,size), 
                        mode='partial',fill_value=np.nan, copy=True).data
        sigma = Cutout2D(fits[f'SIGMA_{band}'].data, position=(x, y), size=(size,size), 
                        mode='partial',fill_value=np.nan, copy=True).data
        mask = Cutout2D(fits[f'MASK_{band}'].data, position=(x, y), size=(size,size), 
                        mode='partial',fill_value=np.nan, copy=True).data
        nparams = fits['FIT_INFO'].data['NFREE'][0]
        chisq = calc_reduced_chisq(data,model,sigma,mask,nparams)
        local_chisq[i] = chisq
        logging.info(f"  reduced chisq: {chisq:.2f}")

    col = pyfits.Column(name='COMP1_CHI2NU',format='E',array=local_chisq)
    header = fits['FINAL_BAND'].header
    hdu = pyfits.BinTableHDU.from_columns(final_band.columns+col,header=header)
    fits['FINAL_BAND'] = hdu

    logging.info(f"Writing {filename}...")
    fits.writeto(filename,overwrite=True)

def draw_residual(ax, data, mask):
    logging.debug("Drawing residual...")
    img = np.ma.array(data,mask=mask,fill_value=-np.inf)
    norm = viz.ImageNormalize(img.compressed(),interval=viz.ZScaleInterval(),clip=False)
    cmap = copy.copy(matplotlib.cm.gray)
    cmap.set_bad('k',1.0)
    ax.imshow(img.filled(),origin='lower',norm=norm,cmap=cmap)

def plot_results(filename,obj=None):
    """Plot an array of results"""

    objid = obj['COADD_OBJECT_ID'] if obj else ''

    names = ['INPUT','MASK','MODEL','RESIDUAL']
    bands = config['bands']

    fits = pyfits.open(filename)
    fig,axes = plt.subplots(len(bands),len(names),figsize=(8,6))
    plt.subplots_adjust(wspace=0.05,hspace=0.05,
                        left=0.05,right=0.98,bottom=0.05,top=0.95)

    for i,(band,name) in enumerate(itertools.product(bands,names)):
        #if isinstance(fits[i],pyfits.BinTableHDU): continue
        extname = name + f'_{band}'
        logging.info(f'{i}: {extname}')

        img = fits[extname].data
        ax = axes.flat[i]
        plt.sca(ax)

        ticks = np.arange(0,img.shape[0],50)[1:-1]
        plt.xticks(ticks,labels=[])
        plt.yticks(ticks,labels=[])
        ax.tick_params(axis='both',direction='in',width=1,length=6,right=True,top=True)

        if name == 'INPUT':
            norm = viz.ImageNormalize(img, interval=viz.ZScaleInterval())

        if name == 'MASK':
            ax.imshow(img,origin='lower',cmap='gray')
        elif name == 'RESIDUAL':
            draw_residual(ax, img, fits[f'MASK_{band}'].data)
        else:
            ax.imshow(img,origin='lower',norm=norm,cmap='gray')

        if (name==names[0]): ax.set_ylabel(band)
        if (band==bands[-1]): ax.set_xlabel(name.lower()[:5])

    plt.suptitle(objid,y=0.98)
    outfile = filename.replace('.fits','.png')
    logging.info("Writing %s..."%outfile)
    plt.savefig(outfile)

def plot_segmap(config,obj,grp):
    objid = obj['COADD_OBJECT_ID']

    cutfile = os.path.join(config['cutdir'],config['cutfile']).format(objid=objid,band='r')
    f = pyfits.open(cutfile)
    img = f['SCI'].data
    seg = f['SEG'].data
    header = f['SEG'].header
    wcs = WCS(header)

    size = img.shape[0] // 1.5

    x,y = wcs.wcs_world2pix(obj['RA'],obj['DEC'],0)
    imgcut = Cutout2D(img, position=(x, y), size=(size,size),wcs=wcs,
                      mode='partial',fill_value=np.nan, copy=True)
    segcut = Cutout2D(seg, position=(x, y), size=(size,size),wcs=wcs,
                      mode='partial',fill_value=np.nan, copy=True)

    fig = plt.figure(figsize=(8,6))

    norm = viz.ImageNormalize(imgcut.data, interval=viz.ZScaleInterval())
    plt.imshow(imgcut.data,origin='lower',norm=norm,cmap='gray')

    color='chartreuse'
    x,y = segcut.wcs.wcs_world2pix(obj['RA'],obj['DEC'],0)
    plt.annotate(obj['COADD_OBJECT_ID'],(x,y),fontsize=12,weight='bold',color=color)
    plt.scatter(x,y,marker='o',s=5,c=color,facecolor='none',edgecolor=color)

    vmin,vmax = None, None

    if grp is not None:
        group = grp[grp['LSBG_COADD_OBJECT_ID']==objid]
        
        vmin = group['OBJECT_NUMBER'].min() - 10
        vmax = group['OBJECT_NUMBER'].max() + 10

        for o in group:
            if o['COADD_OBJECT_ID'] == objid: continue
            x,y = segcut.wcs.wcs_world2pix(o['RA'],o['DEC'],0)
            plt.annotate(o['COADD_OBJECT_ID'],(x,y),fontsize=8,color=color)
            plt.scatter(x,y,marker='x',s=3,c=color)

    array = np.ma.array(segcut.data,mask=segcut.data==0)
    plt.imshow(array,origin='lower',cmap='tab20',vmin=vmin,vmax=vmax,alpha=0.3)
    plt.colorbar(label='OBJECT_NUMBER')

    plt.xlim(0,size)
    plt.ylim(0,size)

    plt.suptitle(objid,y=0.94)

    outfile = os.path.join(config['galdir'],'segmap_{objid}.png').format(objid=objid)
    logging.info("Writing %s..."%outfile)
    plt.savefig(outfile,dpi=100,bbox_inches=None)

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
    if config['grpfile']:
        grp = pyfits.open(config['grpfile'])[1].data.view(np.recarray)
    else:
        grp = None

    objids = np.array(args.objid)

    print("Running galfitm for %i objects..."%len(objids))
    objects = cat[np.in1d(cat['COADD_OBJECT_ID'],objids)]

    for obj in objects:
        objid = obj['COADD_OBJECT_ID']

        galfile = os.path.join(config['galdir'],config['galfile']).format(objid=objid)
        resfile = os.path.join(config['galdir'],config['resfile']).format(objid=objid)
        if True:
            run_galfitm(config,obj)
            subprocess.check_call(f'cp {galfile} {resfile}',shell=True)
            add_bad_mask(config,obj)
            add_sigma_image(config,obj)
            calc_local_chisq(config,obj)

        #if args.plot:
        if True:
            resfile = os.path.join(config['galdir'],config['resfile']).format(objid=objid)
            plot_results(resfile,obj)
            plot_segmap(config,obj,grp)
