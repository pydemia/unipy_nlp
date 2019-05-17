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


def compute_coherence_values(
        dictionary, corpus, id2word, texts,
        num_topic_list=[5, 10],
        lda_typ='default',  # {'default', 'mallet'}
        random_seed=1,
):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    model_list = []
    coherence_list = []

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    if lda_typ == 'default':
        for num_topics in num_topic_list:
            model = gensim.models.LdaMulticore(
                corpus,
                num_topics=num_topics,
                id2word=id2word,
                passes=2,
                workers=8,
                eta='symmetric',
                decay=.8,  # {.5, 1.}
                per_word_topics=False,
                offset=1.,
                iterations=30,
                gamma_threshold=.001,  # 0.001,
                minimum_probability=.05,  # .01,
                minimum_phi_value=.01,
                random_state=1,
            )
            coherence_model = gensim.models.CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            coherence_list += [coherence_model.get_coherence()]

    elif lda_typ == 'hdp':
        for num_topics in num_topic_list:
            model = gensim.models.HdpModel(
                corpus,
                id2word=id2word,
                T=3,
                # alpha=,
                K=num_topics,
                # gamma=,
                # decay=.5, # {.5, 1.}
                # per_word_topics=True,
                # minimum_probability=.1,
                # minimum_phi_value=.01,
                random_state=1,
            )
            coherence_model = gensim.models.CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            coherence_list += [coherence_model.get_coherence()]

    elif lda_typ == 'mallet':
        # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
        mallet_url = 'http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip'
        mallet_filename = mallet_url.split('/')[-1]
        mallet_unzipped_dirname = mallet_filename.split('.zip')[0]
        mallet_path = f'{mallet_unzipped_dirname}/bin/mallet'

        import zipfile
        import urllib

        if not os.path.exists(mallet_path):
            # download the url contents in binary format
            urllib.urlretrieve(mallet_url, mallet_filename)

            # open method to open a file on your system and write the contents
            with zipfile.ZipFile(mallet_filename, "r") as zip_ref:
                zip_ref.extractall(mallet_unzipped_dirname)

        for num_topics in num_topic_list:
            model = gensim.models.wrappers.LdaMallet(
                mallet_path,
                corpus=corpus,
                num_topics=num_topics,
                id2word=id2word,
            )
            coherence_model = gensim.models.CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            # coherence_list += [coherence_model.get_coherence()]

    return model_list, coherence_list


# Print the coherence scores
def pick_best_n_topics(dictionary, corpus, texts, lda_typ='default'):

    model_list, coherence_values = compute_coherence_values(
        dictionary=dictionary,
        corpus=corpus,
        id2word=dictionary,
        texts=texts,
        num_topic_list=[5, 7, 10, 12, 15, 17, 20],
        lda_typ=lda_typ,
        #  start=2, limit=40, step=6,
    )

    paired = zip(model_list, coherence_values)
    ordered = sorted(paired, key=lambda x: x[1], reverse=True)
    best_model = ordered[0][0]

    model_topicnum_list = []
    for i, (m, cv) in enumerate(zip(model_list, coherence_values)):
        topic_num = m.num_topics
        coh_value = round(cv, 4)
        print(
            f'[{i}] Num Topics ({topic_num:2})' +
            f' has Coherence Value of {coh_value}'
        )
        model_topicnum_list += [(topic_num, m)]

    model_dict = dict(model_topicnum_list)
    print(f'Best N topics: {best_model.num_topics}')

    return best_model, model_list, model_dict, coherence_values
