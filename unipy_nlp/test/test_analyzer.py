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
import unipy_nlp.analyzer as uanz

importlib.reload(uprc)
importlib.reload(uanz)


# %% BEFORE STEP: data_collector

prep = uprc.Preprocessor()
prep.read_json('./data/_tmp_dump/rawdata_cpred_flatted.json')
sentence_list = prep.source_sentences

# prep.train_spm(
#     source_type='list',  # {'list', 'txt'}
#     model_type='bpe',
#     vocab_size=30000,
#     model_name='spm_trained',
#     random_seed=1,
# )
prep.load_spm(
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
tokenized = [
    list(filter(lambda w: len(w) > 1, s))
    for s in spmed_unspaced
]

# %%
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

    lda_model, model_list, model_dict, coherence_values = uanz.pick_best_n_topics(
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
print("Test: 'pyLDAvis' finished.")

# %%


print("Test: 'pyLDAvis' finished.")

#%%
