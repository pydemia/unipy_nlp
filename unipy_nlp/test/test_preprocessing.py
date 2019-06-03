"""Test Code Here.
"""

# %%

import elasticsearch as els
from elasticsearch import Elasticsearch
# from hunspell import HunSpell

import os
import re
import json
from glob import glob
from pprint import pprint
import importlib
import pandas as pd
import subprocess
import gensim
import pyLDAvis
import pyLDAvis.gensim as gensimvis

from unicodedata import normalize
import unipy_nlp.data_collector as udcl
import unipy_nlp.preprocessing as uprc

importlib.reload(udcl)
importlib.reload(uprc)


# %% BEFORE STEP: data_collector

# _tmp_df, _tmp_filename = udcl.collect_data(
#     filepath='./data/saveasnew',
#     dump_filepath='./data/_tmp_dump/prep/rawdata_collected.json',
#     dump_json_ok=True,
#     return_tuple=True,
# )

# _tmp_filename = './data/_tmp_dump/prep/rawdata_collected.json'
# _tmp_df = pd.read_json(
#     _tmp_filename,
#     orient='records',
#     encoding='utf-8',
#     lines=True,
# )

# # %% Get Data from Elasticsearch

# ES_HOST = '52.78.243.101'
# ES_PORT = '9200'

# es = Elasticsearch(
#     [
#         {
#             'host': ES_HOST,
#             'port': int(ES_PORT),
#             # 'url_prefix': 'es',
#             'use_ssl': False,
#         },
#     ]
# )

# _tmp_df = uprc.get_data_from_es(
#     es,
#     index='happymap_temp',
#     # match_as_flat_dict={
#     #     'sheet_nm': '17년',
#     #     'table_nm': 'culture_survey',
#     # }
# )
# %%

# sentence_list = _tmp_df['contents'].tolist()


# # %%
# sentence_list[:10]

# #%% Option: Spell Check

# # hunspell_path = './unipy_nlp/_resources/hunspell'
# # spell_checker = HunSpell(
# #     f'{hunspell_path}/ko.dic',
# #     f'{hunspell_path}/ko.aff',
# # )
# # spell_checker.add('수펙스')  # User-Defined Dictionary


# # asent = "무궁화 꽃이 피었읍니다"
# # print(asent)
# # print('spell', [spell_checker.spell(s) for s in asent.split(r' ')])
# # print('suggest', [spell_checker.suggest(s) for s in asent.split(r' ')])
# # print('analyze', [spell_checker.analyze(s) for s in asent.split(r' ')])

# # for _ in sentence_list[:10]:
# #     print(f'udcl: {_}')
# #     print(f'after : {uprc.spell_corrector(_)}\n')
# len(sentence_list)

# # %%
# SPM_MODEL_NAME = 'spm_trained'
# uprc.train_spm(
#     sentence_list,
#     source_type='list',  # {'list', 'txt'}
#     model_type='bpe',
#     vocab_size=50000,
#     model_name=SPM_MODEL_NAME,
#     random_seed=None,
# )

# # %%

# sp = uprc.load_spm(
#     SPM_MODEL_NAME,
# )
# # %%


# spmed = [
#     sp.EncodeAsPieces(l) for l in sentence_list
# ]
# spmed_ids = [
#     sp.EncodeAsIds(l) for l in sentence_list
# ]
# spmed_unspaced = [
#     list(
#         filter(
#             lambda x: len(x) > 1,
#             (t.replace('▁', '') for t in l)
#         )
#     )
#     for l in spmed
# ]

# %% ALL IN ONE (as a `class`)

prep = uprc.Preprocessor()
prep.read_json('./data/_tmp_dump/prep/rawdata_collected.json')
sentence_list = prep.source_sentences

morphed_filtered = prep.pos_tag(
    tag_type=[
        '체언 접두사', '명사', '한자', '외국어',
        '수사', '구분자',
        '동사',
        '부정 지정사', '긍정 지정사',
    ]
)
print(len(morphed_filtered))


prep.train_spm(
    source_type='list',  # {'list', 'txt'}
    model_type='bpe',
    vocab_size=30000,
    model_name='spm_trained',
    savepath='./data/_tmp_dump/spmed',
    random_seed=1,
)
prep.load_spm(
    savepath='./data/_tmp_dump/spmed',
    model_name='spm_trained',
    use_bos=False,
    use_eos=False,
    vocab_min_freq_threshold=None,
)

# %%

prep.spm_encode(
    sentence_list[:3],
    type='piece',
    rm_space=True,
)

# %%
sentence_for_pos_list = [
    "무궁화 꽃이 피었습니다.",
    "우리는 민족중흥의 역사적 사명을 띠고 이 땅에 태어났다.",
]

tokenized_morphed_filtered = prep.pos_tag(
    input_text=sentence_for_pos_list,
    tag_type=[
        '체언 접두사', '명사', '한자', '외국어',
        '수사', '구분자',
        '동사',
        '부정 지정사', '긍정 지정사',
    ]
)
print(tokenized_morphed_filtered)

# %%
pprint([
    "새로운 기술환경의 발전과 확산이 진행되는 it환경",
    "비즈니스 환경과의 접목에 집중해 새로운 사업영역 선점",
])
sentence_for_spm_list = [
    "새로운 기술환경의 발전과 확산이 진행되는 it환경",
    "비즈니스 환경과의 접목에 집중해 새로운 사업영역 선점",
    "장기적인투자 및 성과에 대한 평가 방식 변화 인재영입과 직원 복지 증대",
]

tokenized_spmed = prep.spm_encode(
    sentence_for_spm_list,
    type='piece',
    rm_space=True,
)

pprint(tokenized_spmed)


# %%

list(prep.tagger.tagset.items())
