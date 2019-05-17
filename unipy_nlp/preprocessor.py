# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:14:19 2019

@author: Youngju Jaden Kim
"""


from hunspell import HunSpell
import os
import re
import sys
import json
import random
import warnings
import subprocess
import itertools as it
import functools as ft
import collections
from glob import glob
from pprint import pprint
import numpy as np
import pandas as pd

import gensim
import sentencepiece as spm

import unidecode
from unicodedata import normalize

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


font_dict = {
    path.split('/')[-1][:-4]: path
    for path in fm.get_fontconfig_fonts()
    if 'nanum' in path.lower().split('/')[-1]
}

for _font_nm, _font_file in font_dict.items():
    subprocess.call(['cp', _font_file, './unipy_nlp/_resources/fonts/'])


# %%

# root_path = os.getcwd()
# hunspell_path = './data/_hunspell'
# if not os.path.isdir(hunspell_path):
#     os.makedirs(hunspell_path, exist_ok=False)
#     print(f"'Spell-Dict will be saved in {hunspell_path}")
#     subprocess.call(
#         [
#             'git',
#             'clone',
#             'https://github.com/changwoo/hunspell-dict-ko.git',
#             f'{hunspell_path}'
#         ]
#     )
# os.chdir(hunspell_path)
# subprocess.Popen(
#     'make;',
#     shell=True,
#     stderr=subprocess.STDOUT,
# )
# os.chdir(root_path)

hunspell_path = './unipy_nlp/_resources/hunspell'
spell_checker = HunSpell(
    f'{hunspell_path}/ko.dic',
    f'{hunspell_path}/ko.aff',
)
spell_checker.add('수펙스')  # User-Defined Dictionary


#%%

def spell_corrector(sentence_str):
    splitted = sentence_str.split(r' ')

    if len(splitted) > 0:
        return ' '.join(
            [
                spell_checker.suggest(word)[0]
                if not spell_checker.spell(word)
                else word
                for word in splitted
            ]
        )
    else:
        return sentence_str


# command_train = ' '.join(
#     [
#         # 'spm_train',
#         f'--input={new_file}',
#         f'--model_prefix={SPM_MODEL_NAME}',
#         '' if SPM_MODEL_TYPE == 'word' else f'--vocab_size={SPM_VOCAB_SIZE}',
#         f'--character_coverage=0.9995',
#         # '--seed_sentencepiece_size=10000',
#         # f'--pieces_size={SPM_VOCAB_SIZE}',
#         f'--model_type={SPM_MODEL_TYPE}',
#         f'--input_sentence_size={len(sentenced)}',
#         # f'--max_sentencepiece_length={max(map(len, sentenced))}',
#         f'--max_sentencepiece_length={512}',
#     ],
# )

# %%

def get_data_from_es(
        es_conn_object,
        index,
        match_as_flat_dict=None,
        ) -> pd.DataFrame:

    if match_as_flat_dict is None:
        query_match = {'match_all': {}}
    else:
        query_match = {
            'match': {key: value}
            for key, value in match_as_flat_dict.items()
        }

    query_body = {'query': query_match}
    result_q = es_conn_object.search(
        index=index,
        body=query_body,
    )['hits']['hits']

    return pd.DataFrame(q['_source'] for q in result_q)


# %%

def raw_in_count(filename):
    with open(filename, 'rb') as file:
        bufgen = it.takewhile(
            lambda x: x, (
                file.raw.read(1024*1024)
                for _ in it.repeat(None)
            )
        )
    return sum(buf.count(b'\n') for buf in bufgen)

def train_spm(
        source,
        source_type='list',  # {'list', 'txt'}
        model_type='bpe',
        vocab_size=50000,
        model_name='spm_trained',
        random_seed=None,
        ):
    
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if source_type == 'list':
        spm_source_joined_str = '\n'.join(source)
        spm_source_file = f'./data/_tmp.txt'
        with open(spm_source_file, 'w') as file:
            file.write(spm_source_joined_str)
        input_size_int = len(source)

    elif source_type == 'txt':
        spm_source_file = source
        input_size_int = raw_in_count(spm_source_file)


    command_train = ' '.join(
        [
            # 'spm_train',
            f'--input={spm_source_file}',
            f'--model_prefix={model_name}',
            '' if model_type == 'word' else f'--vocab_size={vocab_size}',
            f'--character_coverage=0.9995',
            f'--hard_vocab_limit={str(False).lower()}',
            # '--seed_sentencepiece_size=10000',
            # f'--pieces_size={SPM_VOCAB_SIZE}',
            f'--model_type={model_type}',
            f'--input_sentence_size={input_size_int}',
            # f'--max_sentencepiece_length={max(map(len, sentenced))}',
            f'--max_sentencepiece_length={512}',
            f'--num_sub_iterations={10}',
            f'--num_threads={16}',
            f'--unk_id=0',
            f'--bos_id=1',
            f'--eos_id=2',
            f'--pad_id=3',
        ],
    )
    spm.SentencePieceTrainer.Train(command_train)


def load_spm(
        model_name,
        ):
    
    sp = spm.SentencePieceProcessor()
    sp.Load(model_name)
    # sp.SetEncodeExtraOptions('eos:bos')
    # sp.LoadVocabulary(
    #     f'{model_name}.vocab',
    #     VOCAB_MIN_THRESHOLD,
    # )
