"""Test Code Here.
"""

# %%

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
import unipy_nlp.preprocessor as uprc
import unipy_nlp.analyzer.topic_modeler as utpm

importlib.reload(uprc)
importlib.reload(utpm)


# %% BEFORE STEP: data_collector

prep = uprc.Preprocessor()
prep.read_json('./data/_tmp_dump/prep/rawdata_cpred_flatted.json')
sentence_list = prep.source_sentences

# %%

morphed_filtered = prep.pos_tag(
    tag_type=[
        '체언 접두사', '명사', '한자', '외국어',
        '수사', '구분자',
        '동사',
        '부정 지정사', '긍정 지정사',
    ]
)
print(len(morphed_filtered))

# %%

# prep.train_spm(
#     source_type='list',  # {'list', 'txt'}
#     model_type='bpe',
#     vocab_size=30000,
#     model_name='spm_trained',
#     savepath = './data/_tmp_dump/spmed',
#     random_seed=1,
# )
prep.load_spm(
    savepath='./data/_tmp_dump/spmed',
    model_name='spm_trained',
    use_bos=False,
    use_eos=False,
    vocab_min_freq_threshold=None,
)

spmed_unspaced = prep.spm_encode(
    sentence_list,
    type='piece',
    rm_space=True,
)
print(len(spmed_unspaced))

# %%
# {nouned, morphed, morphed_filtered, spmed, spmed_unspaced}

# tokenized = [
#     list(filter(lambda w: len(w) > 1, s))
#     for s in spmed_unspaced
# ]
tokenized = [
    list(filter(lambda w: len(w) > 1, s))
    for s in morphed_filtered
]


# %%

tpm = utpm.TopicModeler(sentence_list, tokenized)
tpm.train_lda(
    num_topic=5,
    workers_n=8,
    random_seed=1,
)
tpm.pick_best_lda_topics(
    num_topic_list=[5, 7, 10],
    workers_n=8,
    random_seed=1,
)
tpm.save_lda(savepath='data/_tmp_dump/topic_modeling', affix='lda')
tpm.load_lda('data/_tmp_dump/topic_modeling')
tpm.visualize_lda_to_html(
    7,
    top_n=10,
    r_normalized=False,
    workers_n=8,
    random_seed=1,
    savepath='data/_tmp_dump/topic_modeling',
    filename_affix='lda',
    save_relevent_terms_ok=True,
    save_html_ok=True,
    display_ok=False,
)

sentence_labeled = tpm.estimate_topics_by_documents(
    7,
    # sentence_list=tokenized,
    random_seed=1,
    save_ok=True,
    savepath='data/_tmp_dump/topic_modeling',
    filename_affix='lda',
)
sentence_labeled, topic_freq = tpm.load_estimated(
    target_topic_num=7,
    savepath='data/_tmp_dump/topic_modeling',
    filename_affix='lda',
)

sentence_repr = tpm.get_representitive_documents(
    7,
    len_range=(10, 30),
    top_n=10,
    save_ok=True,
    savepath='data/_tmp_dump/topic_modeling',
    filename_affix='lda',
)
sentence_repr = tpm.load_representitive_documents(
    7,
    top_n=10,
    savepath='data/_tmp_dump/topic_modeling',
    filename_affix='lda',
)

(repr_sentenced,
 repr_bow_corpus_doc,
 repr_bow_corpus_idx) = tpm.get_representitive_candidates(
    len_range=(12, 30),
)

#%%
