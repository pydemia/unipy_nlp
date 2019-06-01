# -*- coding: utf-8 -*-
from __future__ import absolute_import


import os
import re
import io
import json
import sys
import subprocess
import importlib
import pandas as pd

from MeCab import Tagger

from ._backend import (build_mecab, reset_mecabrc,
                       PKG_PATH,
                       MECAB_INSTALLED_DIC_PATH,
                       MECAB_SOURCE_DIC_PATH)

from . import __BUILD_OK__
import sys


__all__ = []
__all__ += [
    'Mecab',
    'build_mecab_user_dic',
]


attrs = ['tags',        # 품사 태그
         'semantic',    # 의미 부류
         'has_jongsung',  # 종성 유무
         'read',        # 읽기
         'type',        # 타입
         'first_pos',   # 첫번째 품사
         'last_pos',    # 마지막 품사
         'original',    # 원형
         'indexed']     # 인덱스 표현

module_installed_path = os.path.dirname(os.path.relpath(__file__))
# dic_installed_path = os.path.join(
#     module_installed_path,
#     '_resources/mecab/mecab/lib/mecab/dic/mecab-ko-dic',
# )
# dic_source_path=os.path.join(
#     module_installed_path,
#     '_resources/mecab/mecab-ko-dic',
# )
# dicpath_to_modulepath = os.path.relpath(
#     module_installed_path,
#     dic_source_path,
# )
module_installed_path = PKG_PATH
dic_installed_path = MECAB_INSTALLED_DIC_PATH
dic_source_path = MECAB_SOURCE_DIC_PATH

def read_json(filename, encoding='utf-8'):
    """JSON file reader."""
    with io.open(filename, 'r', encoding=encoding) as f:
        return json.load(f)


def parse(result, allattrs=False, join=False):
    def split(elem, join=False):
        if not elem:
            return ('', 'SY')
        s, t = elem.split('\t')

        if join:
            return s + '/' + t.split(',', 1)[0]
        else:
            return (s, t.split(',', 1)[0])

    return [split(elem, join=join) for elem in result.splitlines()[:-1]]


class Mecab():
    """Wrapper for MeCab-ko morphological analyzer.

    `MeCab`_, originally a Japanese morphological analyzer and POS tagger
    developed by the Graduate School of Informatics in Kyoto University,
    was modified to MeCab-ko by the `Eunjeon Project`_
    to adapt to the Korean language.

    In order to use MeCab-ko within KoNLPy, follow the directions in
    :ref:`optional-installations`.

    .. code-block:: python
        :emphasize-lines: 1
        >>> from unipy_nlp.tagger import Mecab
        >>> mecab = Mecab()
        >>> print(mecab.morphs(u'영등포구청역에 있는 맛집 좀 알려주세요.'))
        ['영등포구', '청역', '에', '있', '는', '맛집', '좀', '알려', '주', '세요', '.']
        >>> print(mecab.nouns(u'우리나라에는 무릎 치료를 잘하는 정형외과가 없는가!'))
        ['우리', '나라', '무릎', '치료', '정형외과']
        >>> print(mecab.pos(u'자연주의 쇼핑몰은 어떤 곳인가?'))
        [('자연', 'NNG'), ('주', 'NNG'), ('의', 'JKG'), ('쇼핑몰', 'NNG'), ('은', 'JX'), ('어떤', 'MM'), ('곳', 'NNG'), ('인가', 'VCP+EF'), ('?', 'SF')]

    :param dicpath: The path of the MeCab-ko dictionary.

    .. _MeCab: https://taku910.github.io/mecab/
    .. _Eunjeon Project: http://eunjeon.blogspot.kr/
    """
    def __init__(
            self,
            dicpath=dic_installed_path,
            ):

        try:
            self.tagger = Tagger('-d %s' % dicpath)
            self.tagset = read_json(
                '%s/_resources/mecab/mecab_tagset.json' % module_installed_path
            )
        except RuntimeError:
            raise Exception('The MeCab dictionary does not exist at "%s". Is the dictionary correctly installed?\nYou can also try entering the dictionary path when initializing the Mecab class: "Mecab(\'/some/dic/path\')"' % dicpath)
        # except NameError:
        #     raise Exception('Install MeCab in order to use it: http://konlpy.org/en/latest/install/')

    # TODO: check whether flattened results equal non-flattened
    def pos(self, phrase, flatten=True, join=False):
        """POS tagger.

        :param flatten: If False, preserves eojeols.
        :param join: If True, returns joined sets of morph and tag.
        """

        if sys.version_info[0] < 3:
            phrase = phrase.encode('utf-8')
            if flatten:
                result = self.tagger.parse(phrase).decode('utf-8')
                return parse(result, join=join)
            else:
                return [
                    parse(
                        self.tagger.parse(eojeol).decode('utf-8'),
                        join=join,
                    )
                    for eojeol in phrase.split()
                ]

        else:
            if flatten:
                result = self.tagger.parse(phrase)
                return parse(result, join=join)
            else:
                return [
                    parse(
                        self.tagger.parse(eojeol).decode('utf-8'),
                        join=join,
                    )
                    for eojeol in phrase.split()
                ]

    def morphs(self, phrase):
        """Parse phrase to morphemes."""

        return [s for s, t in self.pos(phrase)]

    def nouns(self, phrase):
        """Noun extractor."""

        tagged = self.pos(phrase)
        return [s for s, t in tagged if t.startswith('N')]


def build_mecab_user_dic(nested_list, mode='a'):

    # sample_list = [
    #     ['점심시간', 'T'],
    #     ['워라밸', 'T'],
    #     ['의사 결정', 'T'],
    # ]

    udf_token = pd.DataFrame(
        nested_list,
        columns=['word', 'last_yn'],
    ).drop_duplicates(subset='word')

    udf_token['0'] = udf_token['word']
    udf_token['1'] = 0
    udf_token['2'] = 0
    udf_token['3'] = 1
    udf_token['4'] = 'NNG'
    udf_token['5'] = '*'
    udf_token['6'] = udf_token['last_yn']
    udf_token['7'] = udf_token['word']
    udf_token['8'] = '*'
    udf_token['9'] = '*'
    udf_token['10'] = '*'
    udf_token['11'] = '*'
    udf_token['12'] = '*'

    udf_token_mecab = udf_token.loc[:, udf_token.columns.str.isnumeric()]

    """
        0        1   2    3          4         5        6        7        8        9           10       11      12
    표층형 (표현형태)	좌문맥ID  우문맥ID  출현비용     품사태그    의미부류   종성 유무    읽기      타입    첫번째품사	마지막 품사
    서울              0      0        0         NNG       지명       T      서울       *        *           *         *       *
    불태워졌	         0      0        0    VV+EM+VX+EP     *        T    불태워졌   inflected   VV          EP        *    불태우/VV/+어/EC/+지/VX/+었/EP/
    해수욕장           0      0        0         NNG        *        T    해수욕장   Compound    *           *         *   해수/NNG/+욕/NNG/+장/NNG/*
    """

    udf_token_mecab.to_csv(
        os.path.join(
            dic_source_path,
            'user-dic',
            'udf.csv',
        ),
        mode=mode,
        header=False,
        index=False,
    )

    # commands = "cd {MEACB_DIC_DIR} ;bash ./tools/add-userdic.sh ;cd {PKG_DIR}".format(
    #     MEACB_DIC_DIR=dicpath_to_modulepath,
    #     PKG_DIR=module_installed_path,
    # )

    importlib.reload(__BUILD_OK__)
    if not __BUILD_OK__.__BUILD_OK__:
        print("RE-BUILD 'MeCab' at the first time to add 'user-defined dictionary'.")
        build_mecab()
        importlib.reload(__BUILD_OK__)

    commands = ";".join(
        [
            "cd {MEACB_DIC_DIR}",
            "cp ../add-userdic.sh ./tools",
            "bash ./tools/add-userdic.sh",
        ]
    ).format(
        MEACB_DIC_DIR=dic_source_path,
    )
    out = subprocess.check_output(commands, shell=True)
    print(out.decode('utf-8'))

    reset_mecabrc()
