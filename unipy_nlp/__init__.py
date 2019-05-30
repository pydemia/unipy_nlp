"""NLP Analysis Tools.

This module provides a number of useful functions for natural language handling.

analyzer
------------
- `func` -- Description here.

"""


import os
import sys
import tarfile

from unipy_nlp.__version__ import __version__
from unipy_nlp.__UNIPY_NLP_SETUP__ import __SETUP_OK__
from unipy_nlp import tagger
from unipy_nlp.tagger import *


pkgpath = os.path.dirname(os.path.realpath(__file__))

DIR_EXIST_OK = os.path.isdir(os.path.join(pkgpath, '_resources'))
if not (DIR_EXIST_OK and __SETUP_OK__):
    with tarfile.open(os.path.join(pkgpath, '_resources.tar.gz')) as tfile:
        tfile.extractall(pkgpath)
    
    with open(os.path.join(pkgpath, '__UNIPY_NLP_SETUP__.py'), 'w') as f:
        f.write("__SETUP_OK__ = True")

    print("'_resource' is ready.")


MECAB_BIN_DIR = os.path.join(
    pkgpath,
    '_resources',
    'mecab',
    'bin',
)
sys.path.insert(0, MECAB_BIN_DIR)

__all__ = []
__all__ += ['__version__']

# __all__ += core.__all__
__all__ += tagger.__all__
