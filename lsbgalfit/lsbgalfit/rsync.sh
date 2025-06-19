#!/usr/bin/env bash

# Copy cutouts from from galfit to cutout
#rsync -avh --inplace --include="*/" --filter="-! cut_*.fits" galfit/ cutout/

# Copy pngs from run dir to web
ssh des30.fnal.gov 'cd /home/s1/kadrlica/projects/lsbgalfit/v4/galfit-v0; rsync -avhW --inplace --no-compress ./*/*.png /opt/des-cal/fnalmisc/lsbgalfit/v4'
