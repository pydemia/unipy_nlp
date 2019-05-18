"""Test Code Here.
"""

# %%

import elasticsearch as els
from elasticsearch import Elasticsearch
from hunspell import HunSpell

import os
import re
import json
from glob import glob
import importlib
import pandas as pd
import subprocess
import gensim
import pyLDAvis
import pyLDAvis.gensim as gensimvis

from unicodedata import normalize
import unipy_nlp.data_collector as before
import unipy_nlp.preprocessor as uprc

importlib.reload(before)
importlib.reload(uprc)


# %% BEFORE STEP: data_collector

# _tmp_df, _tmp_filename = uprc.collect_data(
#     './data',
#     dump_json_ok=True,
#     return_tuple=True,
# )
_tmp_filename = './data/_tmp_dump/rawdata_cpred_flatted.json'
_tmp_df = pd.read_json(
    _tmp_filename,
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
    # match_as_flat_dict={
    #     'sheet_nm': '17년',
    #     'table_nm': 'culture_survey',
    # }
)

# %%

sentence_list = _tmp_df['contents'].tolist()


# %%
sentence_list[:10]

#%% Option: Spell Check

# hunspell_path = './unipy_nlp/_resources/hunspell'
# spell_checker = HunSpell(
#     f'{hunspell_path}/ko.dic',
#     f'{hunspell_path}/ko.aff',
# )
# spell_checker.add('수펙스')  # User-Defined Dictionary


# asent = "무궁화 꽃이 피었읍니다"
# print(asent)
# print('spell', [spell_checker.spell(s) for s in asent.split(r' ')])
# print('suggest', [spell_checker.suggest(s) for s in asent.split(r' ')])
# print('analyze', [spell_checker.analyze(s) for s in asent.split(r' ')])

# for _ in sentence_list[:10]:
#     print(f'before: {_}')
#     print(f'after : {uprc.spell_corrector(_)}\n')
len(sentence_list)
#%%
SPM_MODEL_NAME = 'spm_trained'
spm_model = uprc.train_spm(
    sentence_list,
    source_type='list',  # {'list', 'txt'}
    model_type='bpe',
    vocab_size=50000,
    model_name=SPM_MODEL_NAME,
    random_seed=None,
)

# %%

sp = uprc.load_spm(
    SPM_MODEL_NAME,
)
# %%


spmed = [
    sp.EncodeAsPieces(l) for l in sentence_list
]
spmed_ids = [
    sp.EncodeAsIds(l) for l in sentence_list
]
spmed_unspaced = [
    list(
        filter(
            lambda x: len(x) > 1,
            (t.replace('▁', '') for t in l)
        )
    )
    for l in spmed
]

# %%

tokenized = spmed_unspaced  # {nouned, morphed, morphed_filtered, spmed, spmed_unspaced}

# Create Dictionary
cdict = gensim.corpora.Dictionary(tokenized)
cdict.filter_extremes(no_below=30, no_above=.5, keep_n=100000)

max(cdict.keys())

bow_corpus_idx = [cdict.doc2idx(doc) for doc in tokenized]
bow_corpus_raw = [cdict.doc2bow(doc) for doc in tokenized]


tfidf = gensim.models.TfidfModel(bow_corpus_raw)
corpus_tfidf = tfidf[bow_corpus_raw]

bow_corpus_tfidf = bow_corpus_raw  # corpus_tfidf.corpus

bow_corpus = bow_corpus_tfidf
# %%

MODEL_SAVED_OK = False

if not MODEL_SAVED_OK:

    lda_model, model_list, model_dict, coherence_values = uprc.pick_best_n_topics(
        dictionary=cdict,
        corpus=bow_corpus,
        texts=tokenized,
        lda_typ='default',  # 'default',
    )

    for _topic_num, _model in model_dict.items():

        # LDA_TOPIC_NUM = _model.num_topics
        LDA_TOPIC_NUM = _topic_num
        LDA_MODEL_NAME = f'lda_{LDA_TOPIC_NUM}_topics_model.ldamodel'

        print(f'{LDA_TOPIC_NUM:2}: {LDA_MODEL_NAME}')

        _filename = f'data/{LDA_MODEL_NAME}'
        _model.save(_filename)

else:
    n_list = [5, 7, 10, 12, 15, 17, 20]
    model_dict = {}
    model_list = []
    for _topic_num in n_list:

        LDA_TOPIC_NUM = _topic_num
        LDA_MODEL_NAME = f'lda_{LDA_TOPIC_NUM}_topics_model.ldamodel'

        print(f'{LDA_TOPIC_NUM:2}: {LDA_MODEL_NAME}')

        _filename = f'data/{LDA_MODEL_NAME}'

        _model = gensim.models.LdaMulticore.load(_filename)

        model_dict.setdefault(_topic_num, _model)
        model_list += [_model]

# %%

lda_model = model_dict[7]  # 15 is the best
# lda_model = model_list[2] # 15 is the best

LDA_TOPIC_NUM = lda_model.num_topics
LDA_MODEL_NAME = f'happy_lda_{LDA_TOPIC_NUM}topic'
# %%
prepared_data = gensimvis.prepare(
    topic_model=lda_model,
    corpus=bow_corpus,
    dictionary=cdict,
    doc_topic_dist=None,
    R=30,
    lambda_step=0.2,
    mds='tsne',
    # mds=<function js_PCoA>,
    n_jobs=-1,
    plot_opts={'xlab': 'PC1', 'ylab': 'PC2'},
    sort_topics=True,
)

LDA_HTML = f'data/lda_vis_result_{LDA_TOPIC_NUM}_topics.html'
LDA_JSON = f'data/lda_vis_result_{LDA_TOPIC_NUM}_topics.json'

pyLDAvis.save_html(prepared_data, LDA_HTML)
pyLDAvis.save_json(prepared_data, LDA_JSON)

# %%
# pyLDAvis.display(prepared_data, local=False)


# %%
#%% [markdown]
# $$ distinctiveness(w) = \sum P(t \vert w) log\frac{P(t \vert w)}{P(w)} $$
# $$ saliency(w) = P(w) \times distinctiveness(w) $$
#
# <div align="right">(Chuang, J., 2012. Termite: Visualization techniques for assessing textual topic models)</div>

# %%


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


# %%
subprocess

#%%
