"""Backend Support Module.
"""

# %%

import os
import re
import matplotlib.font_manager as fm
import subprocess
import importlib
from . import __BUILD_OK__


__all__ = []
__all__ += [
    'build_mecab',
    'reset_mecabrc',
]
__all__ += [
    '__DIR_EXIST_OK__',
    'PKG_NAME',
    'PKG_PATH',
    'RESOURCE_PATH',
    'FONT_PATH',
    'MECAB_RESOURCE_PATH',
    'MECAB_PATH',
    'MECAB_BIN_DIR',
    'MECAB_LIB_DIR',
    'MECAB_LIBEXEC_DIR',
    'MECAB_INSTALLED_DIC_PATH',
    'MECAB_SOURCE_DIC_PATH',
    'MECABRC_FILE',
    'MECAB_RCDIC_PATH',
]



font_dict = {
    path.split('/')[-1][:-4]: path
    for path in fm.get_fontconfig_fonts()
    if 'nanum' in path.lower().split('/')[-1]
}

for _font_nm, _font_file in font_dict.items():
    subprocess.call(['cp', _font_file, './unipy_nlp/_resources/fonts/'])


# %%

PKG_NAME = 'unipy_nlp'
PKG_PATH = os.path.abspath(os.path.dirname(__file__))

__DIR_EXIST_OK__ = os.path.isdir(os.path.join(PKG_PATH, '_resources'))


RESOURCE_PATH = os.path.join(
    PKG_PATH,
    '_resources',
)
FONT_PATH = os.path.join(
    PKG_PATH,
    '_resources',
    'fonts',
)
MECAB_RESOURCE_PATH = os.path.join(
    PKG_PATH,
    '_resources',
    'mecab',
)

MECAB_PATH = os.path.join(
    MECAB_RESOURCE_PATH,
    'mecab',
)
MECAB_BIN_DIR = os.path.join(
    PKG_PATH,
    '_resources',
    'mecab',
    'bin',
)
MECAB_LIB_DIR = os.path.join(
    MECAB_PATH,
    'lib',
)
MECAB_LIBEXEC_DIR = os.path.join(
    MECAB_PATH,
    'libexec',
    'mecab',
)
MECAB_INSTALLED_DIC_PATH = os.path.join(
    MECAB_PATH,
    'lib/mecab/dic/mecab-ko-dic',
)
MECAB_SOURCE_DIC_PATH = os.path.join(
    RESOURCE_PATH,
    'mecab/mecab-ko-dic',
)

MECABRC_FILE = os.path.join(
    MECAB_PATH,
    'etc',
    'mecabrc',
)
MECAB_RCDIC_PATH = os.path.join(
    MECAB_PATH,
    'dic',
)

def build_mecab():
    output = subprocess.call(
        ['bash', os.path.join(MECAB_RESOURCE_PATH, 'build_mecab.sh')]
    )
    with open(os.path.join(PKG_PATH, '__BUILD_OK__.py'), 'w') as f:
        f.write("__BUILD_OK__ = True")
    importlib.reload(__BUILD_OK__)
    print(output)

def reset_mecabrc():
    with open(MECABRC_FILE, 'r') as rcfile:
        rc_str = rcfile.read()
        replaced = re.sub(r"(dicdir = )(.*)", r"\1" + MECAB_RCDIC_PATH, rc_str)

    with open(MECABRC_FILE, 'w') as rcfile:
        rcfile.write(replaced)

    print(f"'mecabrc' has been changed: '{MECAB_RCDIC_PATH}'")
