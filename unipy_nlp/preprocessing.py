# -*- coding: utf-8 -*-
"""Tokenize text with `sentencepiece` & `MeCab`, from `xlsx` & `Elasticsearch`.
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
    """Text Preprocessing with POS-Tagging or Byte-Pair Encoding.

    Get tokenized from text.

    Parameters
    ----------
    tagger: str `{'mecab',}`
        A POS-Tagging Engine to use.

    Attributes
    ----------
    source_sentences: list
        sentences from text, given by `read_json` or `read_es`.

    Methods
    -------
    read_json

    read_es

    pos_tag

    train_spm

    load_spm

    spm_encode

    See Also
    --------
    POS-Tagging
        ``konlpy.tag.Mecab``

    Byte-Pair Encoding
        ``sentencepiece``

    Examples
    --------
    >>> import unipy_nlp.data_collector as udcl
    >>> import unipy_nlp.preprocessing as uprc
    >>> from pprint import pprint
    >>> prep = uprc.Preprocessor()
    >>> prep.read_json('./data/_tmp_dump/prep/rawdata_collected.json')
    >>> sentence_for_pos_list = [
    ...     "무궁화 꽃이 피었습니다."
    ...     "우리는 민족중흥의 역사적 사명을 띠고 이 땅에 태어났다.",
    ... ]
    >>> tokenized_morphed_filtered = prep.pos_tag(
    ...     input_text=sentence_for_pos_list,
    ...     tag_type=[
    ...         '체언 접두사', '명사', '한자', '외국어',
    ...         '수사', '구분자',
    ...         '동사',
    ...         '부정 지정사', '긍정 지정사',
    ...     ]
    ... )
    >>> print(tokenized_morphed_filtered)
    [['무궁화'], ['우리', '민족중흥', '역사', '사명']]
    >>> prep.train_spm(
    ...     source_type='list',
    ...     model_type='bpe',
    ...     vocab_size=30000,
    ...     model_name='spm_trained',
    ...     savepath='./data/_tmp_dump/spmed',
    ...     random_seed=1,
    ... )
    >>> prep.load_spm(
    ...     savepath='./data/_tmp_dump/spmed',
    ...     model_name='spm_trained',
    ...     use_bos=False,
    ...     use_eos=False,
    ...     vocab_min_freq_threshold=None,
    ... )
    >>> sentence_for_spm_list = [
    ...     "새로운 기술환경의 발전과 확산이 진행되는 it환경",
    ...     "비즈니스 환경과의 접목에 집중해 새로운 사업영역 선점",
    ... ] 
    >>> tokenized_spmed = prep.spm_encode(
    ...     sentence_for_spm_list,
    ...     type='piece',
    ...     rm_space=True,
    ... )
    >>> pprint(tokenized_spmed)
    [['새로운', '기술', '환경의', '발전과', '확산이', '진행되는', 'it', '환경'],
    ['비즈니스', '환경', '과의', '접목', '에', '집중', '해', '새로운', '사업영역', '선점'],

    """
    def __init__(
            self,
            tagger='mecab',  # {'mecab', 'kkma', 'twitter', etc.}
            ):
        self.SPM_MODEL_NAME = None
        self.source_sentences = None

        if tagger == 'mecab':
            self.tagger = Mecab()
            self.tagset = list(self.tagger.tagset.items())

    def drop_by_minimum_length(self, sentence_list, min=2):
        return list(filter(lambda s: len(s) > min, sentence_list))

    def read_json(
            self,
            filename,
            key='contents',
            drop_min=2,
            ):
        """
        Read sentences from disk, as `self.source_sentences`.

        Parameters
        ----------
        filename: str
            A filepath to read.
        
        key: str
            A key of sentences in json object.
        
        drop_min: int (default: 2)
            A lower bond of sentence length.
            If an inappropriate value is given,
            it will be changed by `1` systemically.

        Example
        -------

        >>> import unipy_nlp.preprocessing as uprc
        >>> prep = uprc.Preprocessor()
        >>> prep.read_json(
        ...     './data/_tmp_dump/prep/rawdata_collected.json',
        ...     key='contents',
        ...     drop_min=2,
        ... )
        >>> prep.source_sentences[:2]
        ['새로운 기술환경의 발전과 확산이 진행되는 it환경', '비즈니스 환경과의 접목에 집중해 새로운 사업영역 선점']

        """
        self.data=pd.read_json(
            filename,
            orient='records',
            encoding='utf-8',
            lines=True,
        )

        if drop_min > 1:
            self.source_sentences=self.drop_by_minimum_length(
                    self.data[key].tolist(),
                    min=drop_min,
            )
        else:
            self.source_sentences = self.drop_by_minimum_length(
                    self.data[key].tolist(),
                    min=1,
            )


    def read_es(
            self,
            host,
            port,
            index='happymap_temp',
            match_as_flat_dict=None,
            key='contents',
            drop_min=2,
            ):
        """
        Read sentences from Elasticsearch, as `self.source_sentences`.

        Parameters
        ----------
        host: str
            A domain address of Elasticsearch server.
        
        port: str
            A port number of Elasticsearch server.
        
        index: str
            An index of Elasticsearch server.
        
        match_as_flat_dict: str (default: None)
            An option to `query_match`.
            `match_all` If None.
            Example:
            ```
            match_as_flat_dict={
                'sheet_nm': '2019',
                'table_nm': 'board',
            }
            ```
        key: str
            A key of sentences in an object.

        drop_min: int (default: 2)
            A lower bond of sentence length.
            If an inappropriate value is given,
            it will be changed by `1` systemically.
        
        Example
        -------

        >>> import unipy_nlp.preprocessing as uprc
        >>> ES_HOST = '52.78.243.101'
        >>> ES_PORT = '9200'
        >>> prep = uprc.Preprocessor()
        >>> prep.read_es(
        ...     host=ES_HOST,
        ...     port=ES_PORT,
        ...     index='logs',
        ...     match_as_flat_dict={
        ...         'sheet_nm': '2019',
        ...         'table_nm': 'board',
        ...     },
        ...     key='contents',
        ...     drop_min=2,
        ... )
        >>> prep.source_sentences[:2]
        ['새로운 기술환경의 발전과 확산이 진행되는 it환경', '비즈니스 환경과의 접목에 집중해 새로운 사업영역 선점']

        """
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

        if drop_min > 1:
            self.source_sentences = self.drop_by_minimum_length(
                    self.data[key].tolist(),
                    min=drop_min,
            )
        else:
            self.source_sentences = self.drop_by_minimum_length(
                    self.data[key].tolist(),
                    min=1,
            )

    def pos_tag(
            self,
            input_text=None,
            tag_type=None,
            ):
        """
        POS-Tagging with `input_text` or pre-loaded sentences.

        Parameters
        ----------
        input_text: list (default: `None`)
            A list of sentences.
            If `None`, use `self.source_sentences` internally.
        
        tag_type: list (default: `None`)
            A tag name to subset.
            You can use `'NNP'` or `'일반 명사'` either.


        Return
        ------
        tokenized: list

        Example
        -------

        >>> import unipy_nlp.preprocessing as uprc
        >>> ES_HOST = '52.78.243.101'
        >>> ES_PORT = '9200'
        >>> prep = uprc.Preprocessor()
        >>> sentence_for_pos_list = [
        ...     "무궁화 꽃이 피었습니다.",
        ...     "우리는 민족중흥의 역사적 사명을 띠고 이 땅에 태어났다.",
        ... ]
        >>> tokenized = prep.pos_tag(
        ...     input_text=sentence_for_pos_list,
        ...     tag_type=[
        ...         '체언 접두사', '명사', '한자', '외국어',
        ...         '수사', '구분자',
        ...         '동사',
        ...         '부정 지정사', '긍정 지정사',
        ...         'NNP', 'NNG',
        ...     ]
        ... )
        >>> print(tokenized)
        [['무궁화'], ['우리', '민족중흥', '역사', '사명']]

        """
        if input_text is None:
            input_text = self.source_sentences

        if tag_type is None:
            return [
                self.tagger.pos(s)
                for s in input_text
            ]

        elif tag_type == 'nouns':
            return [
                self.tagger.nouns(s)
                for s in input_text
            ]
        
        elif tag_type == 'morphs':
            return [
                self.tagger.morphs(s)
                for s in input_text
            ]
        
        elif isinstance(tag_type, Iterable):
            # tag_type = tag_list = [
            #     '체언 접두사', '명사', '한자', '외국어',
            #     '수사', '구분자',
            #     '동사',
            #     '부정 지정사', '긍정 지정사',
            # ]
            tag_list = list(tag_type)

            tagset_wanted_from_desc = set([
                tag
                for tag, desc in self.tagger.tagset.items()
                for key in tag_list
                if key in desc
            ])
            tagset_wanted_from_key = set([
                tag
                for tag, desc in self.tagger.tagset.items()
                for key in tag_list
                if key in tag
            ])
            tagset_wanted = list(
                tagset_wanted_from_desc.union(tagset_wanted_from_key)
            )

            return [
                self._get_wanted_morphs(s, tagset_wanted)
                for s in input_text
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
            source_file=None,
            model_type='bpe',
            vocab_size=50000,
            model_name='spm_trained',
            savepath='./data',
            random_seed=None,
            ):
        """
        A high-level wrapper for `sentencepiece.SentencePieceTrainer.Train`.

        Parameters
        ----------
        source_type: str (default: `'list'`)
            `list`: Use `self.source_sentences` as an input.
            `txt`: Use a given text file as an input. It should be split by a sentence.

        model_type: str (default: `'bpe'`, `{'bpe', 'word', 'char', 'unigram'}`)
            A model_type of `sentencepiece`.
        
        vocab_size: int (default: `50000`)
            Embedding size of `sentencepiece`.
        
        model_name: str (default: `'spm_trained'`)
            A filename prefix to save.
        
        savepath: str (default: `'./data'`)
            A dirpath to save.
        
        random_seed: int (default: `None`)
            A random seed number.

        Example
        -------

        >>> import unipy_nlp.preprocessing as uprc
        >>> prep = uprc.Preprocessor()
        >>> prep.read_json('./data/_tmp_dump/prep/rawdata_collected.json')
        >>> prep.train_spm(
        ...     source_type='list',
        ...     model_type='bpe',
        ...     vocab_size=30000,
        ...     model_name='spm_trained',
        ...     savepath='./data/_tmp_dump/spmed',
        ...     random_seed=1,
        ... )

        """

        self.SPM_MODEL_NAME = model_name

        source = self.source_sentences

        if savepath.endswith('/'):
            savepath = savepath[:-1]
        os.makedirs(savepath, exist_ok=True)

        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        if source_type == 'list':
            spm_source_joined_str = '\n'.join(source)
            spm_source_file = os.path.join(
                savepath,
                '_spm_tmp.txt',
            )
            with open(spm_source_file, 'w') as file:
                file.write(spm_source_joined_str)
            input_size_int = len(source)

        elif source_type == 'txt':
            spm_source_file = source_file
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

        os.system(f"mv ./{model_name}.model {savepath}/")
        os.system(f"mv ./{model_name}.vocab {savepath}/")
        os.system(f"rm {spm_source_file}")

    def load_spm(
            self,
            savepath='./data',
            model_name=None,
            use_bos=False,
            use_eos=False,
            vocab_min_freq_threshold=None,
            ):
        """
        A high-level wrapper for `sentencepiece.SentencePieceTrainer.Load`.

        Parameters
        ----------
        savepath: str (default: `'./data'`)
            A dirpath to load.

        model_name: str (default: `'spm_trained'`)
            A filename prefix to load.

        use_bos: bool (default: `False`)
            An option of `SetEncodeExtraOptions`.

        use_eos: bool (default: `False`)
            An option of `SetEncodeExtraOptions`.

        vocab_min_freq_threshold: int (default: `None`)
            An lower bound of vocabulary by its frequency.

        Example
        -------

        >>> import unipy_nlp.preprocessing as uprc
        >>> prep = uprc.Preprocessor()
        >>> prep.read_json('./data/_tmp_dump/prep/rawdata_collected.json')
        >>> prep.load_spm(
        ...     savepath='./data/_tmp_dump/spmed',
        ...     model_name='spm_trained',
        ...     use_bos=False,
        ...     use_eos=False,
        ...     vocab_min_freq_threshold=None,
        ... )

        """

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

        model_filename = os.path.join(
            savepath,
            f'{model_name}.model',
        )
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
        """
        A high-level wrapper for
        `sentencepiece.EncodeAsPieces` or `sentencepiece.EncodeAsIds`.

        Parameters
        ----------
        input_list: list
            A list of sentences to tokenize.

        type: str (default: `'piece'`, `{'piece', 'id'}`)
            Choose encoding type. `'piece': str, 'id': int`

        rm_space: bool (default: `True`)
            An option to remove `"▁"` (U+2581), which represents the whitespace.

        Example
        -------

        >>> import unipy_nlp.preprocessing as uprc
        >>> prep = uprc.Preprocessor()
        >>> prep.read_json('./data/_tmp_dump/prep/rawdata_collected.json')
        >>> prep.load_spm(
        ...     savepath='./data/_tmp_dump/spmed',
        ...     model_name='spm_trained',
        ...     use_bos=False,
        ...     use_eos=False,
        ...     vocab_min_freq_threshold=None,
        ... )
        >>> sentence_for_spm_list = [
        ...     "새로운 기술환경의 발전과 확산이 진행되는 it환경",
        ...     "비즈니스 환경과의 접목에 집중해 새로운 사업영역 선점",
        ... ]
        >>> tokenized_spmed = prep.spm_encode(
        ...     sentence_for_spm_list,
        ...     type='piece',
        ...     rm_space=True,
        ... )
        >>> pprint(tokenized_spmed)
        [['새로운', '기술', '환경의', '발전과', '확산이', '진행되는', 'it', '환경'],
        ['비즈니스', '환경', '과의', '접목', '에', '집중', '해', '새로운', '사업영역', '선점'],

        """
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
