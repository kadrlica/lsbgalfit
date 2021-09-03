#!/usr/bin/env python
"""
Pipeline script for running galfit.
"""
__author__ = "Alex Drlica-Wagner"
import os.path
import logging
import subprocess
import glob

import yaml
import numpy as np

import astropy.io.fits as pyfits

INDEX = """
<html>
  <head>
    <title>DES Y3 LSB Galaxies</title>
</head>
<body>
%(table)s
<br>
%(pages)s
</body>
</html>
"""

TABLE = """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>COADD_OBJECT_ID</th>
    </tr>
  </thead>
  <tbody>
%(rows)s
  </tbody>
"""

ROW = """
    <tr>
      <th>{idx}</th>
      <td><a href="{outfile}">{omin} - {omax}</a></td>
    </tr>  
"""

LIST = """
<ul>
%(rows)s
</ul>
"""

if __name__ == "__main__":
    import parser
    parser = parser.Parser(description=__doc__)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    cat = pyfits.open(config['catfile'])[1].data.view(np.recarray)
    objids = cat['COADD_OBJECT_ID']

    outdir = config['wwwdir']
    if not os.path.exists(outdir): os.makedirs(outdir)

    print("Assembling webpages for %i objects..."%len(cat))
    tablerows = []
    for i,objs in enumerate(np.array_split(cat,len(cat)//1000)):
        imin,imax = i*len(objs),(i+1)*len(objs)
        omin,omax = objs['COADD_OBJECT_ID'][[0,-1]]
        logging.info(f"Processing {imin} - {imax}: {omin} - {omax}...")
        print((i,omin,omax))
        outfile = os.path.join(outdir,f'lsbg_{i:02d}.html')
        cmd = f'www.py {args.config} --outfile {outfile} --imin {imin} --imax {imax}'
        if args.verbose: cmd += '-v'

        logging.debug(cmd)
        subprocess.call(cmd,shell=True)

        row = ROW.format(idx=i,outfile=os.path.basename(outfile),
                         omin=omin,omax=omax)
        tablerows.append(row)

    listrows = []

    # Big
    outbase = 'big.html'
    outfile = os.path.join(outdir,outbase)
    select = "cat['RE_R'] > 40"
    cmd = f'www.py {args.config} --outfile {outfile} -s "{select}"'
    subprocess.call(cmd,shell=True)
    listrows.append(f'<li><a href="{outbase}">{outbase}</a></li>')

    # Small
    outbase = 'small.html'
    outfile = os.path.join(outdir,outbase)
    select = "cat['RE_R'] < 9"
    cmd = f'www.py {args.config} --outfile {outfile} -s "{select}"'
    subprocess.call(cmd,shell=True)
    listrows.append(f'<li><a href="{outbase}">{outbase}</a></li>')

    # chi2nu
    outbase = 'chi2nu.html'
    outfile = os.path.join(outdir,outbase)
    select = "(cat['CHI2NU_G'] > 3) | (cat['CHI2NU_R'] > 3) | (cat['CHI2NU_I'] > 3)"
    cmd = f'www.py {args.config} --outfile {outfile} -s "{select}"'
    subprocess.call(cmd,shell=True)
    listrows.append(f'<li><a href="{outbase}">{outbase}</a></li>')

    # color
    outbase = 'color.html'
    outfile = os.path.join(outdir,outbase)
    select = "((cat['MAG_G'] - cat['MAG_R']) > 1.5) | ((cat['MAG_G'] - cat['MAG_I']) > 1.5)"
    cmd = f'www.py {args.config} --outfile {outfile} -s "{select}"'
    subprocess.call(cmd,shell=True)
    listrows.append(f'<li><a href="{outbase}">{outbase}</a></li>')

    # custom
    outbase = 'custom.html'
    outfile = os.path.join(outdir,outbase)
    select = "cat['CUSTOM'] > 0"
    cmd = f'www.py {args.config} --outfile {outfile} -s "{select}"'
    subprocess.call(cmd,shell=True)
    listrows.append(f'<li><a href="{outbase}">{outbase}</a></li>')

    # mof
    outbase = 'mof.html'
    outfile = os.path.join(outdir,outbase)
    select = "cat['EXTENDED_CLASS_MOF'] == 0"
    cmd = f'www.py {args.config} --outfile {outfile} -s "{select}"'
    subprocess.call(cmd,shell=True)
    listrows.append(f'<li><a href="{outbase}">{outbase}</a></li>')

    # Write the top level index file
    table = TABLE%dict(rows='\n'.join(tablerows))
    pages = LIST%dict(rows='\n'.join(listrows))
    index = INDEX%dict(table=table,pages=pages)

    outfile = os.path.join(outdir,'index.html')
    logging.info(f"Writing {outfile}...")
    with open(outfile,'w') as out:
        out.write(index)
