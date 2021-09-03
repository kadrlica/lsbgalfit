#!/usr/bin/env bash

id=$1;
#id=89653910
username=kadrlica
path=/data/des81.a/data/jsanch87/galfitM_inputs/fitresult
scp ${username}@des51.fnal.gov:${path}/imgblock_${id}.fits .
ds9 -multiframe imgblock_${id}.fits

