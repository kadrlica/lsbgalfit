#!/usr/bin/env python
"""
Pipeline parser
"""
__author__ = "Alex Drlica-Wagner"
import logging
import argparse

class Parser(argparse.ArgumentParser):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('formatter_class',
                          argparse.ArgumentDefaultsHelpFormatter)
        super(Parser,self).__init__(*args,**kwargs)

        self.add_argument('config')
        self.add_argument('-f','--force',action='store_true',
                          help='force overwrite')
        self.add_argument('--imin',type=int,default=None,
                          help='minimum index to run')
        self.add_argument('--imax',type=int,default=None,
                          help='maximum index to run')
        self.add_argument('-o','--objid',nargs='+',type=int,
                          help='specific coadd_object_id')
        self.add_argument('-n','--njobs',default=30,type=int,
                          help='number of jobs to submit')
        self.add_argument('-q','--queue',default='vanilla',
                          help='queue to submit')
        self.add_argument('-v','--verbose',action='store_true',
                           help='output verbosity')

    def _parse_verbose(self,opts):
        """Set logging level based on verbosity"""
        level = logging.DEBUG if vars(opts).get('verbose') else logging.INFO
        logging.getLogger().setLevel(level)

    def parse_args(self,*args,**kwargs):
        opts = super(Parser,self).parse_args(*args,**kwargs)
        self._parse_verbose(opts)
        return opts
