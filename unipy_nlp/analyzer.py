# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:41:19 2019

@author: Youngju Jaden Kim
"""

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
import pyLDAvis
import pyLDAvis.gensim as gensimvis

import unidecode
from unicodedata import normalize

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def compute_coherence_values(
        dictionary,
        corpus,
        id2word,
        texts,
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
def pick_best_n_topics(
        dictionary,
        corpus,
        texts,
        lda_typ='default',
        ):
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


def get_saliency(tinfo_df):
    """Calculate Saliency for terms within a topic.

    $$ distinctiveness(w) = \sum P(t \vert w) log\frac{P(t \vert w)}{P(w)} $$
    $$ saliency(w) = P(w) \times distinctiveness(w) $$
    <div align="right">(Chuang, J., 2012. Termite: Visualization techniques for assessing textual topic models)</div>

    Parameters
    ----------
    tinfo: pandas.DataFrame
        `pyLDAvis.gensim.prepare`.to_dict()['tinfo'] containing
        ['Category', 'Freq', 'Term', 'Total', 'loglift', 'logprob']

    """

    saliency = tinfo_df['Freq'] / tinfo_df['Total']

    return saliency


def get_relevance(tinfo_df, l=.6):
    """Calculate Relevances with a given lambda value.

    # $$ relevance(t,w) = \lambda \cdot P(w \vert t) + (1 - \lambda) \cdot \frac{P(w \vert t)}{P(w)} $$
    <div align="center"> Recommended $\lambda = 0.6$ </div>
    <div align="right">(Sievert, C., 2014. LDAvis: A method for visualizing and interpreting topics)</div>

    Parameters
    ----------
    tinfo: pandas.DataFrame
        `pyLDAvis.gensim.prepare`.to_dict()['tinfo'] containing
        ['Category', 'Freq', 'Term', 'Total', 'loglift', 'logprob']
    
    l: float
        lambda_ratio between {0-1}. default is .6 (recommended from its paper)

    """

    relevance = l * tinfo_df['logprob'] + (1 - l) * tinfo_df['loglift']

    return relevance


def groupby_top_n(
        dataframe,
        group_by=None,
        order_by=None,
        ascending=False,
        n=5,
        ):

    res_df = (
        dataframe
        .groupby(group_by)
        [dataframe.columns.drop(group_by)]
        .apply(
            lambda x: x.sort_values(order_by, ascending=ascending).head(n)
        )
    )
    return res_df


def _df_with_names(data, index_name, columns_name):
    """A renaming function from `pyLDAvis._prepare`.
    """
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _series_with_name(data, name):
    """A renaming function from `pyLDAvis._prepare`.
    """
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)


def get_terminfo_table(
        lda_model,
        corpus: list = None,
        dictionary: gensim.corpora.dictionary.Dictionary = None,
        doc_topic_dists=None,
        use_gensim_prepared=True,
        top_n=10,
        r_normalized=False,
        random_seed=1,
        ):

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    if use_gensim_prepared:

        _prepared = gensimvis.prepare(
            topic_model=lda_model,
            corpus=corpus,
            dictionary=dictionary,
            doc_topic_dist=None,
            R=len(dictionary),
            # lambda_step=0.2,
            mds='tsne',
            # mds=<function js_PCoA>,
            n_jobs=-1,
            plot_opts={'xlab': 'PC1', 'ylab': 'PC2'},
            sort_topics=True,
        )
        tinfo_df = pd.DataFrame(_prepared.to_dict()['tinfo'])

        tinfo_df['topic_term_dists'] = np.exp(tinfo_df['logprob'])
        tinfo_df['term_proportion'] = (
            np.exp(tinfo_df['logprob']) / np.exp(tinfo_df['loglift'])
        )
        tinfo_df['saliency'] = get_saliency(tinfo_df)
        tinfo_df['relevance'] = get_relevance(tinfo_df)

        tinfo_df['term_prob'] = np.exp(tinfo_df['logprob'])
        tinfo_df['term_r_prob'] = np.exp(tinfo_df['relevance'])
        tinfo_df['term_r_adj_prob'] = (
            tinfo_df
            .groupby(['Category'])
            ['term_r_prob']
            .apply(lambda x: x / x.sum())
        )

        if r_normalized:
            r_colname = 'term_r_adj_prob'
        else:
            r_colname = 'term_r_prob'

        relevance_score_df = (
            tinfo_df[tinfo_df['Category'] != 'Default']
            .groupby(['Category', 'Term'])
            [[r_colname]]
            .sum()
            .reset_index()
        )

        # corpus_dict_df = pd.DataFrame(
        #     # It is possible
        #     # because the keys of this dictionary generated from range(int).
        #     # Usually the dictionary is iterable but not ordered.
        #     list(dictionary.values()),
        #     # [dictionary[i] for i, _ in enumerate(dictionary)],
        #     columns=['Term'],
        # )
        # corpus_dict_df['term_id'] = corpus_dict_df.index
        corpus_dict_df = pd.DataFrame(
            list(dictionary.items()),
            columns=['term_id', 'Term'],
        )
        corpus_dict_df.set_index('term_id', drop=False, inplace=True)

        r_score_df = pd.merge(
            relevance_score_df,
            corpus_dict_df,
            on=['Term'],
            how='left',
        )
        r_score_df['category_num'] = (
            r_score_df['Category']
            .str
            .replace('Topic', '')
            .astype(int) - 1
        ).astype('category')
        r_score_df.set_index(['category_num', 'term_id'], inplace=True)
        ixs = pd.IndexSlice

        topic_list = r_score_df.index.levels[0]
        equal_prob = 1. / len(topic_list)
        empty_bow_case_list = list(
            zip(topic_list, [equal_prob] * len(topic_list))
        )

        def get_bow_score(
                bow_chunk,
                score_df=r_score_df,
                colname=r_colname,
        ):

            bow_chunk_arr = np.array(bow_chunk)
            word_id_arr = bow_chunk_arr[:, 0]
            word_cnt_arr = bow_chunk_arr[:, 1]

            # normed_word_cnt_arr = (word_cnt_arr / word_cnt_arr.sum()) * 10
            clipped_word_cnt_arr = np.clip(word_cnt_arr, 0, 3)

            score_series = (score_df.loc[ixs[:, word_id_arr], :]
                            .groupby(level=0)
                            [colname]
                            .apply(lambda x: x @ clipped_word_cnt_arr)
                            )
            score_list = list(score_series.iteritems())
            # normed_score_series = score_series / score_series.sum()
            # score_list = list(normed_score_series.iteritems())

            return score_list

        bow_score_list = [
            get_bow_score(bow_chunk)
            if bow_chunk not in (None, [])
            else empty_bow_case_list
            for bow_chunk in corpus
        ]

        relevant_terms_df = groupby_top_n(
            tinfo_df,
            group_by=['Category'],
            order_by=['relevance'],
            ascending=False,
            n=top_n,
        )
        relevant_terms_df['rank'] = (
            relevant_terms_df
            .groupby(['Category'])
            ['relevance']
            # .rank(method='max')
            .rank(method='max', ascending=False)
            .astype(int)
        )

    else:
        vis_attr_dict = gensimvis._extract_data(
            topic_model=ldamodel,
            corpus=corpus,
            dictionary=dictionary,
            doc_topic_dists=None,
        )
        topic_term_dists = _df_with_names(
            vis_attr_dict['topic_term_dists'],
            'topic', 'term',
        )
        doc_topic_dists = _df_with_names(
            vis_attr_dict['doc_topic_dists'],
            'doc', 'topic',
        )
        term_frequency = _series_with_name(
            vis_attr_dict['term_frequency'],
            'term_frequency',
        )
        doc_lengths = _series_with_name(
            vis_attr_dict['doc_lengths'],
            'doc_length',
        )
        vocab = _series_with_name(
            vis_attr_dict['vocab'],
            'vocab',
        )

        ## Topic
        # doc_lengths @ doc_topic_dists
        topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()
        topic_proportion = (topic_freq / topic_freq.sum())

        ## reorder all data based on new ordering of topics
        # topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
        # topic_order = topic_proportion.index
        # topic_freq = topic_freq[topic_order]
        # topic_term_dists = topic_term_dists.iloc[topic_order]
        # doc_topic_dists = doc_topic_dists[topic_order]

        # token counts for each term-topic combination
        term_topic_freq = (topic_term_dists.T * topic_freq).T
        term_frequency = np.sum(term_topic_freq, axis=0)

        ## Term
        term_proportion = term_frequency / term_frequency.sum()

        # compute the distinctiveness and saliency of the terms
        topic_given_term = topic_term_dists / topic_term_dists.sum()
        kernel = (topic_given_term *
                  np.log((topic_given_term.T / topic_proportion).T))
        distinctiveness = kernel.sum()
        saliency = term_proportion * distinctiveness

        default_tinfo_df = pd.DataFrame(
            {
                'saliency': saliency,
                'term': vocab,
                'freq': term_frequency,
                'total': term_frequency,
                'category': 'default',
                'logprob': np.arange(len(vocab), 0, -1),
                'loglift': np.arange(len(vocab), 0, -1),
            }
        )

        log_lift = np.log(topic_term_dists / term_proportion)
        log_prob = log_ttd = np.log(topic_term_dists)

    return tinfo_df, relevant_terms_df, r_score_df, bow_score_list


# def get_repr_tokenized(repr_sentenced, dictionary=None):

#     def get_wanted_morphs(s, wanted_tags):
#         res_pos = tagger.pos(s)

#         res = list(
#             filter(
#                 lambda x: (x[1] in wanted_tags) and (len(x[0]) > 1),
#                 res_pos,
#             )
#         )
#         return [morph[0] for morph in res]

#     morphed_filtered = [
#         get_wanted_morphs(s, tagset_wanted)
#         for s in repr_sentenced
#     ]

#     spm_source = (
#         # sentenced
#         [' '.join(s) for s in morphed_filtered]
#         # [''.join(s) for s in morphed]
#     )
#     spm_source_joined = '\n'.join(
#         spm_source
#     )

#     SPM_VOCAB_SIZE = 50000
#     SPM_MODEL_TYPE = 'word'  # {unigram (default), bpe, char, word}
#     SPM_MODEL_NAME = f'happy_spm_{SPM_MODEL_TYPE}_{SPM_VOCAB_SIZE}'

#     random.seed(1)
#     np.random.seed(1)

#     sp = spm.SentencePieceProcessor()
#     sp.Load(f'{SPM_MODEL_NAME}.model')
#     # sp.EncodeAsPieces
#     # sp.EncodeAsIds
#     # sp.DecodePieces
#     # sp.NBestEncodeAsPieces

#     spmed = [
#         sp.EncodeAsPieces(l) for l in spm_source
#     ]
#     spmed_ids = [
#         sp.EncodeAsIds(l) for l in spm_source
#     ]
#     spmed_unspaced = [
#         list(
#             filter(
#                 lambda x: len(x) > 1,
#                 (t.replace('▁', '') for t in l)
#             )
#         )
#         for l in spmed
#     ]
#     bow_corpus_idx = [dictionary.doc2idx(doc) for doc in tokenized]
#     bow_corpus_raw = [dictionary.doc2bow(doc) for doc in tokenized]

#     return spmed_unspaced, bow_corpus_raw, bow_corpus_idx

# (tokenized_repr,
#  bow_corpus_raw_repr,
#  bow_corpus_idx_repr) = get_repr_tokenized(
#     representitive_short_sentences_df,
#     dictionary=cdict,
# )
