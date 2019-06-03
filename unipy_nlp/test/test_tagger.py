"""Test Code Here.
"""


# %%

from unipy_nlp.tagger import Mecab
import MeCab
# %%

import os
import re
import json
from glob import glob
import importlib
import pandas as pd
import subprocess
from unicodedata import normalize
import gensim
import pyLDAvis
import pyLDAvis.gensim as gensimvis

import unipy_nlp.preprocessing as uprc
import unipy_nlp.tagger as utag

from unipy_nlp.tagger import Mecab
from unipy_nlp.tagger import build_mecab_user_dic as build_dic
from unipy_nlp._backend import build_mecab

importlib.reload(uprc)
importlib.reload(utag)


#%%

sentenced = [
    "무궁화 꽃이 피었습니다."
    "우리는 민족중흥의 역사적 사명을 띠고 이 땅에 태어났다.",
    "수펙스협의회에서 self design 상을 캔미팅에서 받았다.",
]

tagger = utag.Mecab()


tag_list = [
    '체언 접두사', '명사', '한자', '외국어',
    '수사', '구분자',
    '동사',
    '부정 지정사', '긍정 지정사',
]
tagset_wanted = [
    tag
    for tag, desc in tagger.tagset.items()
    for key in tag_list
    if key in desc
]
nouned = [
    tagger.nouns(s)
    for s in sentenced
]
morphed = [
    tagger.morphs(s)
    for s in sentenced
]


def get_wanted_morphs(s, wanted_tags):
    res_pos = tagger.pos(s)

    res = list(
        filter(
            lambda x: (x[1] in wanted_tags) and (len(x[0]) > 1),
            res_pos,
        )
    )
    return [morph[0] for morph in res]


morphed_filtered = [
    get_wanted_morphs(s, tagset_wanted)
    for s in sentenced
]

# %%


sample_list = [
    ['점심시간', 'T'],
    ['워라밸', 'T'],
    ['의사 결정', 'T'],
]
build_dic(sample_list)

build_mecab()
