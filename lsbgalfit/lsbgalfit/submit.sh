#!/usr/bin/env bash
# Submit jobs through multiple screen sessions

# Scripts
cmd="python code/run_00.0_all.py"
#cmd="python code/run_00.0_all.py -r feedme -r galfit"
#cmd="python code/run_01.0_cutout.py"
#cmd="python code/run_02.0_feedme.py"
#cmd="python code/run_03.0_galfitm.py"

# Options
#opts='-f -n 50'
opts='-n 50'

# Indexes
imin=
imax=

# Job submission loop
#for imax in $(seq -w 2000 1000 24000); do
#for imax in $(seq -w 0000 4000 24000); do
for imax in $(seq -w 0000 1000 15000); do
    # Skip first entry
    [ -z $imin ] && { imin=$imax; continue; }
    echo "Submit $imin - $imax ..."
    # Submit chunk
    \screen -S sub${imin} -d -m bash -c "$cmd config.yaml --imin $imin --imax $imax $opts"
    #\screen -S sub${imin} -d -m bash -c "sleep 1000s"
    # Iterate
    imin=$imax
done

screen -ls
