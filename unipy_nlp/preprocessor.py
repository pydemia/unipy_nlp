# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:14:19 2019

@author: Youngju Jaden Kim
"""


import elasticsearch as els
from elasticsearch import Elasticsearch
# from hunspell import HunSpell
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
from collections.abc import Iterable

import gensim
import sentencepiece as spm

import unidecode
from unicodedata import normalize

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# from konlpy.tag import Mecab
from .tagger import Mecab


__all__ = []
__all__ += [
    # 'get_data_from_es',
    'Preprocessor',
    # 'get_wanted_morphs',
]


font_dict = {
    path.split('/')[-1][:-4]: path
    for path in fm.get_fontconfig_fonts()
    if 'nanum' in path.lower().split('/')[-1]
}

for _font_nm, _font_file in font_dict.items():
    subprocess.call(['cp', _font_file, './unipy_nlp/_resources/fonts/'])


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
    count_q = es_conn_object.search(
        index=index,
        body=query_body,
    )['hits']['total']
    result_q = es_conn_object.search(
        index=index,
        body=query_body,
        size=count_q,
    )['hits']['hits']

    return pd.DataFrame(q['_source'] for q in result_q)


def get_wanted_morphs(s, wanted_tags):
        tagger = Mecab()
        res_pos = tagger.pos(s)

        res = list(
            filter(
                lambda x: (x[1] in wanted_tags) and (len(x[0]) > 1),
                res_pos,
            )
        )
        return [morph[0] for morph in res]

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

    else:
        raise TypeError(
            "`source_type` should be one of `{'list', 'txt'}`."
        )

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

    os.system(f'rm {spm_source_file}')


def load_spm(
        model_name,
        use_bos=False,
        use_eos=False,
        vocab_min_freq_threshold=None,
        ):
    model_filename = f'{model_name}.model'
    sp = spm.SentencePieceProcessor()
    sp.Load(model_filename)

    if use_bos:
        sp.SetEncodeExtraOptions('bos')
    if use_eos:
        sp.SetEncodeExtraOptions('eos')
    if vocab_min_freq_threshold is not None:
        sp.LoadVocabulary(
            f'{model_name}.vocab',
            vocab_min_freq_threshold,
        )

    return sp


def spm_encode(
        spm_object,
        input_list,
        type='piece',  # {'id', 'piece'}
        rm_space=True,
        ):
    if type == 'piece':
        spmed = [
            spm_object.EncodeAsPieces(l)
            for l in input_list
        ]
    elif type == 'id':
        spmed = [
            spm_object.EncodeAsIds(l)
            for l in input_list
        ]
    else:
        raise TypeError(
            "`input_list` should be one of `{'list', 'txt'}`."
        )

    if rm_space:
        spmed = [
            list(
                filter(
                    lambda x: len(x) > 1,
                    (t.replace('▁', '') for t in l)
                )
            )
            for l in spmed
        ]

    return spmed


class Preprocessor(object):

    def __init__(
            self,
            tagger='mecab',  # {'mecab', 'kkma', 'twitter', etc.}
            ):
        self.SPM_MODEL_NAME = None
        self.source_sentences = None
        self.tagger = Mecab()

    def drop_by_minimum_length(self, sentence_list, min=2):
        return list(filter(lambda s: len(s) > min, sentence_list))

    def read_json(
            self,
            filename,
            drop_min = 2,
            ):

        self.data=pd.read_json(
            filename,
            orient = 'records',
            encoding = 'utf-8',
            lines = True,
        )

        if drop_min:
            self.source_sentences=self.drop_by_minimum_length(
                    self.data['contents'].tolist(),
                    min = drop_min,
            )
        else:
            self.source_sentences = self.drop_by_minimum_length(
                    self.data['contents'].tolist(),
                    min=1,
            )


    def read_es(
            self,
            host,
            port,
            index='happymap_temp',
            match_as_flat_dict=None,
            drop_min=2,
            ):
        es = Elasticsearch(
            [
                {
                    'host': host,
                    'port': int(port),
                    'use_ssl': False,
                },
            ]
        )

        self.data = get_data_from_es(
            es,
            index=index,
            match_as_flat_dict=match_as_flat_dict,
        )

        if drop_min:
            self.source_sentences = self.drop_by_minimum_length(
                    self.data['contents'].tolist(),
                    min=drop_min,
            )
        else:
            self.source_sentences = self.drop_by_minimum_length(
                    self.data['contents'].tolist(),
                    min=1,
            )

    def pos_tag(
            self,
            tag_type=None,
            ):
        """
        tag_type
        
        tag_type = tag_list = [
            '체언 접두사', '명사', '한자', '외국어',
            '수사', '구분자',
            '동사',
            '부정 지정사', '긍정 지정사',
        ]
        """

        if tag_type is None:
            return [
                self.tagger.pos(s)
                for s in self.source_sentences
            ]

        elif tag_type == 'nouns':
            return [
                self.tagger.nouns(s)
                for s in self.source_sentences
            ]
        
        elif tag_type == 'morphs':
            return [
                self.tagger.morphs(s)
                for s in self.source_sentences
            ]
        
        elif isinstance(tag_type, Iterable):
            # tag_type = tag_list = [
            #     '체언 접두사', '명사', '한자', '외국어',
            #     '수사', '구분자',
            #     '동사',
            #     '부정 지정사', '긍정 지정사',
            # ]
            tag_list = list(tag_type)

            tagset_wanted = [
                tag
                for tag, desc in self.tagger.tagset.items()
                for key in tag_list
                if key in desc
            ]

            return [
                self._get_wanted_morphs(s, tagset_wanted)
                for s in self.source_sentences
            ]

    def _get_wanted_morphs(self, s, wanted_tags):
        res_pos = self.tagger.pos(s)

        res = list(
            filter(
                lambda x: (x[1] in wanted_tags) and (len(x[0]) > 1),
                res_pos,
            )
        )
        return [morph[0] for morph in res]


    def train_spm(
            self,
            source_type='list',  # {'list', 'txt'}
            model_type='bpe',
            vocab_size=50000,
            model_name='spm_trained',
            random_seed=None,
            ):

        self.SPM_MODEL_NAME = model_name

        source = self.source_sentences

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

        else:
            raise TypeError(
                "`source_type` should be one of `{'list', 'txt'}`."
            )

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

        os.system(f'rm {spm_source_file}')

    def load_spm(
            self,
            model_name=None,
            use_bos=False,
            use_eos=False,
            vocab_min_freq_threshold=None,
            ):
        if model_name is None:
            if self.SPM_MODEL_NAME is None:
                raise AttributeError(
                    ' '.join([
                        'An inappropriate `model_name` is given.',
                        'Call `train_spm` to train a new spm model or',
                        'input a proper value on `model_name`.',
                    ])
                )
            else:
                model_name = self.SPM_MODEL_NAME
        else:
            self.SPM_MODEL_NAME = model_name

        model_filename = f'{model_name}.model'
        sp = spm.SentencePieceProcessor()
        sp.Load(model_filename)

        if use_bos:
            sp.SetEncodeExtraOptions('bos')
        if use_eos:
            sp.SetEncodeExtraOptions('eos')
        if vocab_min_freq_threshold is not None:
            sp.LoadVocabulary(
                f'{model_name}.vocab',
                vocab_min_freq_threshold,
            )

        self.spm_model = sp

    def spm_encode(
            self,
            input_list,
            type='piece',  # {'id', 'piece'}
            rm_space=True,
            ):

        spm_model = self.spm_model
        if type == 'piece':
            spmed = [
                spm_model.EncodeAsPieces(l)
                for l in input_list
            ]
        elif type == 'id':
            spmed = [
                spm_model.EncodeAsIds(l)
                for l in input_list
            ]
        else:
            raise TypeError(
                "`input_list` should be one of `{'list', 'txt'}`."
            )

        if rm_space:
            spmed = [
                [t.replace('▁', '') for t in l]
                for l in spmed
            ]
            # spmed = [
            #     list(
            #         filter(
            #             lambda x: len(x) > 1,
            #             (t.replace('▁', '') for t in l)
            #         )
            #     )
            #     for l in spmed
            # ]

        return spmed
