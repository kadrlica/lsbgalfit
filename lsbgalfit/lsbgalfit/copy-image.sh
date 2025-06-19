#!/usr/bin/env bash

# i-band
#rsync -avzh /data/des71.a/data/kwei/data/DES-Y3-tiles/data/ image/
# g,r-band
rsync --dry-run -avzh --exclude 'config*' /data/des71.b/data/kwei/data/DES-Y3-tiles/data/ image/

rsync --dry-run -avzh --exclude 'config*' /data/des71.b/data/kwei/data/DES-Y3-tiles/data/DES2359* image/

#dirname=/data/des71.b/data/kwei/data/DES-Y3-tiles/data
#for dir in $(\ls -d $dirname/DES0000-*); do
#    tilename=$(basename $dir)
#    outdir=image/$tilename
#    echo csub rsync -avzh $dir/ $outdir/
#done;