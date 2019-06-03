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
import unipy_nlp.preprocessing as uprc
import unipy_nlp.analyze.word2vec as uw2v

importlib.reload(uprc)
importlib.reload(uw2v)


# %% BEFORE STEP: data_collector

prep = uprc.Preprocessor()
prep.read_json('./data/_tmp_dump/rawdata_cpred_flatted.json')
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


# # %%
# prep.train_spm(
#     source_type='list',  # {'list', 'txt'}
#     model_type='bpe',
#     vocab_size=30000,
#     model_name='spm_trained',
#     random_seed=1,
# )
# prep.load_spm(
#     model_name='spm_trained',
#     use_bos=False,
#     use_eos=False,
#     vocab_min_freq_threshold=None,
# )

# spmed_unspaced = prep.spm_encode(
#     sentence_list,
#     type='piece',
#     rm_space=True,
# )
# print(len(spmed_unspaced))

# %%
# {nouned, morphed, morphed_filtered, spmed, spmed_unspaced}
tokenized = [
    list(filter(lambda w: len(w) > 1, s))
    for s in morphed_filtered
]

# %%

w2v = uw2v.Word2Vec(tokenized)
w2v.train_w2v(
    size=70,
    window=4,
    min_count=10,
    negative=16,
    workers=8,
    iter=50,
    sg=1,
)
w2v.save_w2v('data/_tmp_dump/word2vec/tmp_word2vec.w2v')
w2v.load_w2v('data/_tmp_dump/word2vec/tmp_word2vec.w2v')

# %%

w2v.get_similar(
    ['구성원', '행복', '필요', '중요'],
    top_n=10,
)

# %%
w2v.save_tensorboard('data/_tmp_dump/tsboard')


#%%
