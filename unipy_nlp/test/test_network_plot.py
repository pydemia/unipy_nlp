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
import unipy_nlp.network_plot as unet

importlib.reload(uprc)
importlib.reload(utpm)
importlib.reload(unet)


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

tokenized = [
    list(filter(lambda w: len(w) > 1, s))
    for s in morphed_filtered
]


# %%

tpm = utpm.TopicModeler(sentence_list, tokenized)
# tpm.train_lda(
#     num_topic=5,
#     workers_n=8,
#     random_seed=1,
# )
# tpm.pick_best_lda_topics(
#     num_topic_list=[5, 7, 10],
#     workers_n=8,
#     random_seed=1,
# )
# tpm.save_lda(savepath='data/_tmp_dump', affix='lda')
tpm.load_lda('data/_tmp_dump')

tpm.lda_model_dict.keys()

# %%

tpm.visualize_lda_to_html(
    7,
    top_n=10,
    r_normalized=False,
    workers_n=6,
    random_seed=1,
    savepath='data/_tmp_dump/topic_modeling',
    filename_affix='lda',
    # save_type='html',  # {'html', 'json'}
    save_relevent_terms_ok=True,
    save_html_ok=True,
    display_ok=False,
)

# sentence_labeled, topic_freq = tpm.estimate_topics_by_documents(
#     7,
#     # sentence_list=tokenized,
#     random_seed=1,
#     save_ok=True,
#     savepath='data/_tmp_dump/topic_modeling',
#     filename_affix='lda',
# )
sentence_labeled, topic_freq = tpm.load_estimated(
    target_topic_num=7,
    savepath='data/_tmp_dump/topic_modeling',
    filename_affix='lda',
)
# sentence_repr = tpm.get_representitive_documents(
#     7,
#     len_range=(10, 30),
#     top_n=10,
#     save_ok=True,
#     savepath='data/_tmp_dump/topic_modeling',
#     filename_affix='lda',
# )
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

# %%
# repr_sentences_df = (
#     pd.Series(
#         list(filter(
#             lambda x: 5 <= len(x) < 25,
#             sentence_list,
#         )),
#         name='documents',
#     )
#     .to_frame()
# )
# repr_sentenced = repr_sentences_df['documents'].tolist()
# repr_sentenced = [
#     r for r in repr_sentenced
#     if 10 <= len(r) < 25
# ]
# repr_bow_corpus_doc = [cdict.doc2bow(doc) for doc in tokenized_repr]

# tpm.corpora_dict
# %%
tpm.topic_freq_df
tpm.top_relevant_terms_df
tpm.bow_corpus_doc
tpm.corpora_dict

# %%

importlib.reload(unet)
vnet = unet.WordNetwork(
    topic_freq_df=tpm.topic_freq_df,
    top_relevant_terms_df=tpm.top_relevant_terms_df,
)

# vnet.get_ngram(tokenized)
# vnet.save_ngram('data/_tmp_dump/network_plot/ngram.json', type='json')
# vnet.save_ngram('data/_tmp_dump/network_plot/ngram.csv', type='csv')

vnet.load_ngram('data/_tmp_dump/network_plot/ngram.json', type='json')
# vnet.load_ngram('data/_tmp_dump/network_plot/ngram.csv', type='csv')


vnet.draw(
    height="100%",
    width='80%',  # "100%",
    bgcolor='#ffffff',  # "#222222",
    font_color='black',  # "white",
    directed=True,
    topic_top_n=5,
    node_freq_threshold=100,
    show_buttons=True,
)

vnet.save('data/_tmp_dump/network_plot/vnet.html')


# %%
(score_dict,
 score_dict_indiced) = vnet.get_topic_mutuality_score_dict(
     cdict=tpm.corpora_dict
)
core_repr = vnet.get_network_scored_repr_docs(
    bow_corpus=repr_bow_corpus_doc,
    repr_docs=repr_sentenced,
    save_ok=True,
    filepath=None,
)

core_repr

#%%
