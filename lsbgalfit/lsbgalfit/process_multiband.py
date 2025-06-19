import os
import udg
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import udg.utils.utils as utils
from udg.utils.utils import time_func
from udg.lsst.detectionwrapper import *
from astropy.io import fits
from astropy.wcs import WCS
from lsst.geom import Box2I, Box2D, Point2I, Point2D, Extent2I, Extent2D
import glob
from astropy.nddata import Cutout2D
import astropy.table
from time import time
@time_func
def get_bbox(catalogs, candidates, tilepath, bands, searchradius=6.0):
    """
    sets up necessary files for Galfit to run

    input:
        catalogs: dict
          Dictionary containing lsst.afw.table.source.source.SourceCatalog
          objects for each of the bands
        coadds: dict
          Dictionary containing MakeCoadd objects for each of the bands
    """
    catalog_out = dict() # Placeholder catalog that will serve to join all the individual-band catalogs
    targetmask_out = dict()
    for band in bands: 
        catalog = catalogs[band] 
        # get footprint centroids and IDs of objects that need to be removed
        fpxy, fpwh = utils.GetFootprintPosition(catalog)
        fpmax, fpmin = utils.GetFootprintIndices(catalog)
        #print(tilepath.split('/')[-2].split('DES')[1])
        candidates_this = candidates[candidates['TILENAME']=='DES'+tilepath.split('/')[-2].split('DES')[1]].copy()
        #print(len(candidates_this), len(candidates), 'Candidates')
        targetra = candidates_this['RA']
        targetdec = candidates_this['DEC']
        targetradec = (targetra, targetdec)
        #print(tilepath, 'tilepath')
        #print(fits.getheader(tilepath, ext=1))
        wcs = WCS(fits.getheader(tilepath, ext=1))
        targetxy = wcs.all_world2pix(targetra, targetdec, 0)
        fpradec = wcs.all_pix2world(fpxy[0], fpxy[1], 0)
        # get peak locations
        peakxy, peak2fp = utils.GetPeakPosition(catalog)
        peakradec = wcs.all_pix2world(peakxy[0], peakxy[1], 0)
        print('\nusing %.2f" search radius'%(searchradius))
        _, peakidx, targetidx, matchdis = utils.FilterByTarget(
                                                        peakradec,
                                                        targetradec,
                                                        radius=searchradius) 
        # set up targetmask for oroginal detection catalog
        targetmask = np.zeros_like(fpradec[0]).astype(bool)
        for _peakidx in peakidx:
            targetmask[peak2fp[_peakidx]] = True
        targetmask_out[band] = targetmask
        if len(targetidx)!=len(targetra):
            allidxset = set(range(0, len(targetra)))
            for _idx in (allidxset - set(targetidx)):
                missingid = candidates.iloc[_idx].COADD_OBJECT_ID
                print('COADD #%s !NOT! processed'%(missingid))
        maxy = np.zeros(np.count_nonzero(targetmask))
        maxx = np.zeros_like(maxy)
        minx = np.zeros_like(maxx)
        miny = np.zeros_like(minx)
        coaddid = np.zeros(np.count_nonzero(targetmask), dtype=candidates['COADD_OBJECT_ID'].dtype)
        stackid = np.zeros(np.count_nonzero(targetmask), dtype=catalog['id'].dtype)
        # Get all the bounding boxes for all bands and LSBGs
        for i, source in enumerate(catalog[targetmask]):
            candidate_i = targetidx[i]
            catalog_i = source['id']-1 # the index for the original detection catalog
            stackid[i] = source['id']-1
            peaki = peakidx[i]
            candidate = candidates_this.iloc[candidate_i]
            coaddid[i] = candidate['COADD_OBJECT_ID']
            # calculate box given footprint position and width and height
            xbounds, ybounds = utils.boxcalculation((peakxy[0][peaki], peakxy[1][peaki]),
                                            (fpwh[0][catalog_i], fpwh[1][catalog_i]))
            _minx, _maxx = xbounds
            _miny, _maxy = ybounds
            _maxx+=1
            _maxy+=1
            minx[i] = _minx; maxx[i] = _maxx; miny[i] = _miny; maxy[i] = _maxy 
        catalog_out[band] = astropy.table.Table([coaddid, stackid, fpxy[0][targetmask_out[band]], fpxy[1][targetmask_out[band]], 
                                minx, maxx, miny, maxy],
                                names=('coaddId', 'stackId', 'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax'))
    joint_cat = catalog_out[bands[0]]
    for i in range(len(bands)-1):
        joint_cat = astropy.table.join(joint_cat, catalog_out[bands[i+1]],
                        keys='coaddId', join_type='outer', table_names=[bands[i], bands[i+1]]) 
        #print(joint_cat.keys())
    joint_cat[f'xmin_{bands[-1]}']=joint_cat['xmin']
    joint_cat[f'ymin_{bands[-1]}']=joint_cat['ymin']
    joint_cat[f'xmax_{bands[-1]}']=joint_cat['xmax']             
    joint_cat[f'ymax_{bands[-1]}']=joint_cat['ymax']           
    return joint_cat, targetmask_out, wcs

def cut_and_clean(coadds, catalogs, targetmask_out, joint_cat, wcs, bands, tilepath):
    # Compute the bounding boxes
    minx = []; miny = []; maxx = []; maxy = [];
    t0 = time()
    for row in joint_cat:
            miny.append(np.nanmin([row[f'ymin_{band}'] for band in bands]))
            minx.append(np.nanmin([row[f'xmin_{band}'] for band in bands]))
            maxy.append(np.nanmax([row[f'ymax_{band}'] for band in bands]))
            maxx.append(np.nanmax([row[f'xmax_{band}'] for band in bands]))
    t1 = time()
    #print('Time to get bounding boxes', t1-t0)
    for band in bands:
        coadd = coadds[band]
        catalog = catalogs[band]
        targetmask = targetmask_out[band]
        fpxy, fpwh = utils.GetFootprintPosition(catalog)
        fpmax, fpmin = utils.GetFootprintIndices(catalog)
        t0 = time()
        # Substitute the footprints of all non candidates with noise
        # the good thing is that if something is not detected, it is not replaced by noise
        clean_coadd = LSSTClean(catalog['id'][~targetmask]-1, catalog, coadd)
        t1 = time()
        print('Time to clean coadd', t1-t0)
        for i, row in enumerate(joint_cat):
            # copy detection mask
            mask = coadd.mask.array[int(miny[i]):int(maxy[i]), int(minx[i]):int(maxx[i])].copy()
            catalog_i = row['stackId']
            coaddid = row['coaddId']
            #print('CoaddID', coaddid, 'Band', band)
            img = Cutout2D(clean_coadd.image.array, (0.5*(minx[i]+maxx[i]), 0.5*(miny[i]+maxy[i])),
                            (maxy[i]-miny[i], maxx[i]-minx[i]), wcs=wcs, copy=True)
            # save cleaned cutout to disk
            hdu = fits.PrimaryHDU(img.data)
            hdu.header.update(img.wcs.to_header())
            t0 = time()
            hdu.writeto(os.path.join(args.out_path, 'image', f'image_{coaddid}_{band}.fits'),
                    overwrite=True)
            ## save PSF to disk
            sampleBBox = Box2I(Point2I(minx[i], miny[i]), Extent2I(maxx[i]-minx[i], maxy[i]-miny[i]))
            subset = coadd[sampleBBox].clone()
            subset.setPsf(coadd.getPsf())
            cutoutpsf = subset.getPsf()
            hdu2 = fits.PrimaryHDU(cutoutpsf.computeImage(Point2D(0.5*(minx[i]+maxx[i]), 0.5*(miny[i]+maxy[i]))).array)
            hdu2.writeto(os.path.join(args.out_path, 'psf', f'psf_{coaddid}_{band}.fits'),
                    overwrite=True)
    return minx, maxx, miny, maxy

def build_feedme_one(coaddid, minx, miny, maxx, maxy, bands, candidates, wcs):        
    # construct feedme
    with open('./baseM.feedme', 'r') as myfile:
        basefeedme = myfile.read()
    imagepath_base = os.path.join(args.out_path, 'image', f'image_{coaddid}')
    psfpath_base = os.path.join(args.out_path, 'psf', f'psf_{coaddid}')
    candidate_mask = np.where(candidates['COADD_OBJECT_ID']==coaddid)[0][0]
    mags = [candidates[f'MAG_AUTO_{band.upper()}'][candidate_mask] for band in bands]
    #print('mags', mags)
    radii = [candidates[f'FLUX_RADIUS_{band.upper()}'][candidate_mask] for band in bands]
    #print('radii', radii)
    bands_out = ''
    imagepath = ''
    psfpath = ''
    magout = ''
    reffout = ''
    for i, band in enumerate(bands[:-1]):
        imagepath += imagepath_base+f'_{band}.fits,'
        bands_out += f'{band},'
        psfpath += psfpath_base+f'_{band}.fits,'
        magout += f'{mags[i]},'
        reffout += f'{radii[i]},'
    imagepath += imagepath_base+f'_{bands[-1]}.fits'
    bands_out += f'{bands[-1]}'
    psfpath += psfpath_base+f'_{bands[-1]}.fits'
    magout += f'{mags[-1]}'
    reffout += f'{radii[-1]}'
    outputpath = os.path.join(args.out_path, 'fitresult', 'imgblock_%d.fits'%coaddid)
    #constraintpath = os.path.join(args., 'feedme/INITIAL-FIT.CONSTRAINTS')
    xmax = maxx-minx
    ymax = maxy-miny
    DESx, DESy = wcs.all_world2pix(candidates['RA'][candidate_mask], candidates['DEC'][candidate_mask], 0)
    positionx = int(DESx-minx)
    positiony = int(DESy-miny)
    ellip = candidates['B_IMAGE'][candidate_mask]/candidates['A_IMAGE'][candidate_mask]
    feedme = basefeedme.format(imagepath=imagepath,
                               bands_out=bands_out,
                               outputpath=outputpath, 
                               psfpath=psfpath, 
                               xmax=xmax, ymax=ymax,
                               positionx=positionx, positiony=positiony,
                               magout=magout, reff=reffout,
                               ellip=ellip)
    # save feedme to disk
    output_file = open(os.path.join(args.out_path,
                       'feedme', 'galfit_%d.feedme'%coaddid), 'w')
    output_file.write(feedme)
    output_file.close()

def LSSTClean(removeids, catalog, coadd):
    from lsst.meas.base import NoiseReplacer, NoiseReplacerConfig
    # construct fp of all objects that need to be removed
    fp_dict = {}
    for idx in removeids:
        measRecord = catalog[int(idx)]
        fp_dict[measRecord.getId()] = (
                                        measRecord.getParent(),
                                        measRecord.getFootprint()
                                    )
    coaddcopy = coadd.clone()
    # instantiate NoiseReplacer. show it our calexp and footprints
    nr_config = NoiseReplacerConfig()
    noiseReplacer = NoiseReplacer(nr_config, coaddcopy, fp_dict)
    return coaddcopy

def main(args):
    # check if tile has been processed
    candidates = pd.read_csv(args.candidate_path)
    #print('Candidates read')
    tiledictpath = '/data/des60.b/data/jsanch87/udg/udg/galfit/tile_dict_v3.pkl'
    tiledict = pickle.load(open(tiledictpath, 'rb'))
    #print('loaded dictionary')
    #if os.path.isfile('tilenames.npy')==False:
    #    tilenames = glob.glob(os.path.join(args.base_dir,'*','*_g.fits.fz'))
    #    np.save('tilenames', np.array(tilenames))
    #else:
    tilenames = np.load('tilenames.npy')
    tilenumbers = np.array([tilename.split('/')[-2] for tilename in tilenames])
    tilemask = np.in1d(tilenumbers, candidates['TILENAME'])
    print('Total number of ``valid`` tiles,', np.count_nonzero(tilemask))
    processed = []
    processed_list = glob.glob('/data/des81.a/data/jsanch87/galfitM_inputs/feedme/*.feedme')
    for fname in processed_list:
        processed.append(int(fname.split('/')[-1].split('.')[0].split('_')[1]))
    processed_candidates = np.in1d(candidates['COADD_OBJECT_ID'], processed)
    processed_tiles = candidates['TILENAME'][processed_candidates]
    unprocessed_tiles = candidates['TILENAME'][~np.in1d(candidates['TILENAME'], processed_tiles)]
    tilemask = np.in1d(tilenumbers, unprocessed_tiles)
    print('Total number of tiles to go,', np.count_nonzero(tilemask))
    final_tile = min(args.tile_fin, len(tilenames[tilemask])-1)
    for tilename in tilenames[tilemask][args.tile_init:final_tile]:
        tilenumber = tilename.split('/')[-2]
        if (tilename in tiledict.keys()):
            print('%s has been processed, skipping...'%tilename)
            continue
        elif tilenumber in tiledict.keys():
            print('%s has been processed, skipping...'%tilenumber)
            continue
        else:
            # perform makefeedme pipeline
            print('\nprocessing %s...'%tilename)
            coadds = dict()
            catalogs = dict()
            for band in args.bands:
                if band=='r':
                    coaddpath = tilename.replace('g.fits.fz', 'r.fits.fz')
                if band=='i':
                    coaddpath = tilename.replace(args.base_dir, '/data/des71.a/data/kwei/data/DES-Y3-tiles/data')
                    coaddpath = coaddpath.replace('g.fits.fz', 'i.fits.fz')
                if band=='g':
                    coaddpath = tilename
                coadds[band] = udg.lsst.detectionwrapper.MakeCoadd(coaddpath)
                catalogs[band] = udg.lsst.detectionwrapper.SimpleDetection(coadds[band])
            joint_catalog, targetmask, wcs =  get_bbox(catalogs, candidates, tilename, args.bands)
            minx, maxx, miny, maxy = cut_and_clean(coadds, catalogs, targetmask, joint_catalog, wcs, args.bands, coaddpath)
            t0 = time()
            [build_feedme_one(joint_catalog['coaddId'][i], minx[i], miny[i], maxx[i], maxy[i], args.bands, candidates, wcs) for i in range(len(joint_catalog))]
            t1 = time()
            print('Generated', len(joint_catalog), 'feedme files in', t1-t0, 'seconds')
            print(joint_catalog['coaddId'])
            # write tilename
            tiledict[tilename] = 1
            with open(tiledictpath, 'wb') as f:
                pickle.dump(tiledict, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # sets up argparser
    parser = argparse.ArgumentParser(description='Process and generate GalFitM feedme files for DES LSBGs')
    parser.add_argument('--base-dir', type=str, default='/data/des71.b/data/kadrlica/data/DES-Y3-tiles/data',
                       help='Path to the original files with the coadd tiles', dest='base_dir')
    parser.add_argument('--initial-tile', type=int, default=0, dest='tile_init', help='Initial tile in the tile-list to process')
    parser.add_argument('--final-tile', type=int, default=100, dest='tile_fin', help='Final tile in the tile-list to process')
    parser.add_argument('--bands', type=list, default=['g','r','i'], help='Bands to process')
    parser.add_argument('--radius', nargs='?', const=1, type=float, default=6.0)
    parser.add_argument('--candidate-path', type=str, dest='candidate_path', default='/data/des71.b/data/kwei/data/DES-Y3-tiles/LSBG_candidates.csv',
        help='Path to candidates file')
    parser.add_argument('--out-path', type=str, default='/data/des81.a/data/jsanch87/galfitM_inputs/', dest='out_path'
    , help='Path to input files for GalFit-M fits')
    args = parser.parse_args()

    main(args)
