"""Test Code Here.
"""

# %%

import elasticsearch as els
from elasticsearch import Elasticsearch
import os
import re
from glob import glob
import importlib
import pandas as pd
import subprocess
from unicodedata import normalize
import unipy_nlp.data_collector as before
import unipy_nlp.preprocessor as uprc
from hunspell import HunSpell

importlib.reload(before)
importlib.reload(uprc)


# %% BEFORE STEP: data_collector

# _tmp_df = before.collect_data('./data', dump_json_ok=False)
_tmp_df = pd.read_json(
    './data/_tmp_dump/rawdata_cpred_flatted.json',
    orient='records',
    encoding='utf-8',
    lines=True,
)

# %% Get Data from Elasticsearch

ES_HOST = '52.78.243.101'
ES_PORT = '9200'

es = Elasticsearch(
    [
        {
            'host': ES_HOST,
            'port': int(ES_PORT),
            # 'url_prefix': 'es',
            'use_ssl': False,
        },
    ]
)

_tmp_df = uprc.get_data_from_es(
    es,
    index='happymap_temp',
    match_as_flat_dict={
        'sheet_nm': '17년',
        'table_nm': 'culture_survey',
    }
)

sentence_list = _tmp_df['contents'].tolist()


# %%
sentence_list[:10]

#%% Option: Spell Check

hunspell_path = './unipy_nlp/_resources/hunspell'
spell_checker = HunSpell(
    f'{hunspell_path}/ko.dic',
    f'{hunspell_path}/ko.aff',
)
spell_checker.add('수펙스')  # User-Defined Dictionary


asent = "무궁화 꽃이 피었읍니다"
print(asent)
print('spell', [spell_checker.spell(s) for s in asent.split(r' ')])
print('suggest', [spell_checker.suggest(s) for s in asent.split(r' ')])
print('analyze', [spell_checker.analyze(s) for s in asent.split(r' ')])

for _ in sentence_list[:10]:
    print(f'before: {_}')
    print(f'after : {uprc.spell_corrector(_)}\n')

#%% 

spm_source_joined_str = '\n'.join(sentence_list)
spm_source_file = f'./data/full_sentence.txt'
with open(spm_source_file, 'w') as file:
    file.write(spm_source_joined_str)

SPM_VOCAB_SIZE = 50000
SPM_MODEL_TYPE = 'word'  # {unigram (default), bpe, char, word}
SPM_MODEL_NAME = f'happy_spm_{SPM_MODEL_TYPE}_{SPM_VOCAB_SIZE}'

command_train = ' '.join(
    [
        # 'spm_train',
        f'--input={new_file}',
        f'--model_prefix={SPM_MODEL_NAME}',
        '' if SPM_MODEL_TYPE == 'word' else f'--vocab_size={SPM_VOCAB_SIZE}',
        f'--character_coverage=0.9995',
        # '--seed_sentencepiece_size=10000',
        # f'--pieces_size={SPM_VOCAB_SIZE}',
        f'--model_type={SPM_MODEL_TYPE}',
        f'--input_sentence_size={len(sentenced)}',
        # f'--max_sentencepiece_length={max(map(len, sentenced))}',
        f'--max_sentencepiece_length={512}',
    ],
)
#%%

subprocess
pkg_list = ['g++', 'openjdk-7-jdk', 'python-dev', 'python3-dev']
#%%
os.getcwd()
# %%
pkg_list = ['g++', 'openjdk-7-jdk', 'python-dev', 'python3-dev']
os.system(
    ';'.join([
        'cd ./unipy_nlp/_resources/pkgs',
        *[f'apt-get download {pkg}' for pkg in pkg_list],
        'cd ../../../',
    ])
)
# os.system('cd ./unipy_nlp/_resources/pkgs')


#%%
