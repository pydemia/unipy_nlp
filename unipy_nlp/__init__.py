"""NLP Analysis Tools.

unipy_nlp
=========
Provides
  1. NLP Data Handling Tools
  2. NLP Analysis Functions.
  3. POS-Tagger (MeCab, non-sudo intall required.)
  4. Topic Modeling(LDA)
  5. Word2Vec
  6. NLP network plot.

How to use
----------
In terms of Data science, Data Preprocessing & Plotting is one of the most
annoying parts of Data Analysis. ``unipy`` offers you many functions maybe
once you have tried to search in ``google`` or ``stackoverflow``.

The docstring examples assume that `unipy` has been imported as `up`::
  >>> import unipy_nlp as unlp

Use the built-in ``help`` function to view a function's docstring::
  >>> help(np.sort)
  ... # doctest: +SKIP

General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``docs`` sub-module::
  >>> from unipy import docs
  >>> help(docs)
  ... # doctest: +SKIP

Available subpackages
---------------------
data_collector
    Get Data from `xlsx`.

tagger
    A MeCab Wrapper without installation.

preprocessing
    Tokenize text with `sentencepiece` & `MeCab`, from `xlsx` & `Elasticsearch`.

analyze
    Topic Modeling(LDA) & Word2Vec.

network_plot
    A N-gram network plot.

test
    Test-codes of ``unipy_nlp``

This module provides a number of useful functions for natural language handling.

analyze
------------
- `func` -- Description here.

"""


import re
import os
import sys
import tarfile
import subprocess
from ._backend import build_mecab

from .__version__ import __version__
from .__UNIPY_NLP_SETUP__ import __SETUP_OK__
from ._backend import (__DIR_EXIST_OK__,
                                PKG_NAME,
                                PKG_PATH,
                                RESOURCE_PATH,
                                FONT_PATH,
                                MECAB_RESOURCE_PATH,
                                MECAB_PATH,
                                MECAB_BIN_DIR,
                                MECAB_LIB_DIR,
                                MECAB_LIBEXEC_DIR,
                                MECAB_INSTALLED_DIC_PATH,
                                MECAB_SOURCE_DIC_PATH,
                                MECABRC_FILE,
                                MECAB_RCDIC_PATH)

from . import _backend
from . import data_collector
from . import tagger
from . import preprocessing
from . import analyze

# from unipy_nlp.tagger import *


# pkgpath = os.path.dirname(os.path.realpath(__file__))

# __DIR_EXIST_OK__ = os.path.isdir(os.path.join(pkgpath, '_resources'))
if not (__DIR_EXIST_OK__ and __SETUP_OK__):
    with tarfile.open(os.path.join(PKG_PATH, '_resources.tar.gz')) as tfile:
        tfile.extractall(PKG_PATH)
    
    # build_mecab()
    with open(os.path.join(PKG_PATH, '__UNIPY_NLP_SETUP__.py'), 'w') as f:
        f.write("__SETUP_OK__ = True")

    # MECAB_PATH = os.path.join(
    #     pkgpath,
    #     '_resources',
    #     'mecab',
    #     'mecab',
    # )
    # MECAB_DICPATH = (
    #     MECAB_PATH,
    #     'dic',
    # )
    # mecabrc_file = os.path.join(
    #     MECAB_PATH,
    #     'etc',
    #     'mecabrc',
    # )

    tagger.reset_mecabrc()
    # with open(MECABRC_FILE, 'r') as rcfile:
    #     rc_str = rcfile.read()
    #     replaced=re.sub(r"^dicdir = (.*)$", MECAB_RCDIC_PATH, rc_str)

    # with open(MECABRC_FILE, 'w') as rcfile:
    #     rcfile.write(replaced)

    # sed - i - e "s/^dicdir = (.*)$/$MECABRC_DICPATH/g" $MECAB_DIR/etc/mecabrc
    print("'_resource' is ready.")


# MECAB_BIN_DIR = os.path.join(
#     pkgpath,
#     '_resources',
#     'mecab',
#     'bin',
# )
sys.path.insert(0, MECAB_BIN_DIR)

# MECAB_LIB_DIR = os.path.join(
#     pkgpath,
#     '_resources',
#     'mecab',
#     'lib',
# )
# MECAB_LIBEXEC_DIR = os.path.join(
#     pkgpath,
#     '_resources',
#     'mecab',
#     'libexec',
#     'mecab',
# )
os.environ['LD_LIBRARY_PATH'] = ':'.join(
    [
        f'{MECAB_LIB_DIR}',
        f'{MECAB_LIB_DIR}',
        os.environ.get('LD_LIBRARY_PATH', ""),
    ]
)

__all__ = []
__all__ += ['__version__']

# __all__ += core.__all__
__all__ += _backend.__all__
__all__ += data_collector.__all__
__all__ += tagger.__all__
__all__ += preprocessor.__all__
__all__ += analyzer.__all__

