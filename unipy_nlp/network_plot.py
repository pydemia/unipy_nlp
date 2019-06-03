# -*- coding: utf-8 -*-
"""A N-gram network plot.
"""

import os
import re
import sys
import json
import zipfile
import urllib
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

from wordcloud import WordCloud
import gensim
from gensim.corpora import Dictionary as corpora_dict
import sentencepiece as spm
from pyvis.network import Network as net
from sklearn.preprocessing import minmax_scale

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm

from .tagger import Mecab


def skip_gram_pairs(
        token_ids,
        window_size=2,
        num_grams=4,
        # negative_samples=1.,
        shuffle=False,
        ):

    res_list = []
    for i, xt in enumerate(token_ids):

        # if i < window_size:
        #     xt_f = token_ids[max(0, i-window_size):i]
        # else:
        #     xt_f = token_ids[i-window_size:i]
        xt_f = token_ids[max(0, i-window_size):i]

        xt_b = token_ids[i+1:i+1+window_size]

        x_source = [xt]
        x_target = xt_f + xt_b

        try:
            x_skipped = np.random.choice(
                x_target,
                min(len(x_target), num_grams),
                replace=False,
            )
        except ValueError as err:
            # print(
            #     'x_source: %s' % x_source,
            #     'x_target: %s' % x_target,
            #     'xt: %s' % xt,
            #     'xt_f: %s' % xt_f,
            #     'xt_b: %s' % xt_b,
            #     sep='\n',
            # )
            # break
            """
            Case: `xt` Only.
            x_source: ['7']
            x_target: []
            xt: 7
            xt_f: []
            xt_b: []
            x_source: ['70924']
            x_target: []
            xt: 70924
            xt_f: []
            xt_b: []
            x_source: ['2659']
            x_target: []
            xt: 2659
            xt_f: []
            xt_b: []
            """
            continue

        res_list += list(it.product(x_source, x_skipped))

    if shuffle:
        np.random.shuffle(res_list)

    return res_list


def feature_map(arr):

        print(arr.shape)
        feature = {
            'target': arr[0],
            'context': arr[1],
        }
        res = pd.DataFrame.from_dict(
            feature,
        )
        return res


def sg_pair_list(
        line_token_ids,
        window_size=5,
        num_grams=6,
        ):

    paired_list = skip_gram_pairs(
        line_token_ids,
        window_size=window_size,
        num_grams=num_grams,
        # negative_samples=1.,
        shuffle=False,
    )

    return paired_list


def sg_pair_array(
        line_token_ids,
        window_size=5,
        num_grams=6,
        ):

    paired_arr = np.array(
        skip_gram_pairs(
            line_token_ids,
            window_size=window_size,
            num_grams=num_grams,
            # negative_samples=1.,
            shuffle=False,
        ),
        # dtype=np.int64,
        dtype=np.str,
    )
    # arr_shape = paired_arr.shape
    # print(paired_arr[:, 0])

    return paired_arr


def token_pair_extractor(token_ls):
    token_ids = token_ls
    print('line length: ', len(token_ids))
    example_array_list = [
        sg_pair_list(
            line_token_ids,
            window_size=4,
            num_grams=8,
        )
        for line_token_ids in token_ids
    ]
    res = sum(example_array_list, [])
    return res


# %% Graph

class WordNetwork(object):
    """A network plot of co-occurance of words.

    Parameters
    ----------

    topic_freq_df: list
        A rank table by topic frequency.

    top_relevant_terms_df: list
        A rank table of `Category`.

    Attributes
    ----------
    pyvis_net: `pyvis.network.Network`

    ngramed_list: `list`

    ngramed_df: `pandas.DataFrame`

    Methods
    -------

    get_ngram

    save_ngram

    load_ngram

    draw

    get_topic_mutuality_score_dict

    get_network_scored_repr_docs


    See Also
    --------

    Preprocessing
        ``unipy_nlp.preprocessing.Preprocessor``
    
    Topic Modeling
        ``unipy_nlp.analyze.topic_modeling.Topic_modeler``

    POS-Tagging
        ``konlpy.tag.Mecab``

    Byte-Pair Encoding
        ``sentencepiece``

    Examples
    --------

    >>> import unipy_nlp.data_collector as udcl
    >>> import unipy_nlp.preprocessing as uprc
    >>> import unipy_nlp.analyze.topic_modeling as utpm
    >>> import unipy_nlp.network_plot as unet
    >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
    >>> tpm.train_lda(...)
    >>> tpm.visualize_lda_to_html(...)
    >>> vnet = unet.WordNetwork(
    ...     topic_freq_df=tpm.topic_freq_df,
    ...     top_relevant_terms_df=tpm.top_relevant_terms_df,
    ... )
    >>> vnet.get_ngram(tokenized)
    >>> vnet.save_ngram('data/_tmp_dump/network_plot/ngram.json', type='json')
    >>> vnet.save_ngram('data/_tmp_dump/network_plot/ngram.csv', type='csv')
    >>> vnet.load_ngram('data/_tmp_dump/network_plot/ngram.json', type='json')
    >>> vnet.load_ngram('data/_tmp_dump/network_plot/ngram.csv', type='csv')
    >>> vnet.draw(
    ...     height="100%",
    ...     width='800px',
    ...     bgcolor='#ffffff',
    ...     font_color='black',
    ...     directed=True,
    ...     topic_top_n=5,
    ...     node_freq_threshold=100,
    ...     show_buttons=True,
    ... )
    >>> (score_dict,
    ... score_dict_indiced) = vnet.get_topic_mutuality_score_dict(
    ...     cdict=tpm.corpora_dict
    ... )
    >>> core_repr = vnet.get_network_scored_repr_docs(
    ...     bow_corpus=repr_bow_corpus_doc,
    ...     repr_docs=repr_sentenced,
    ...     save_ok=True,
    ...     savepath=None,
    ... )

    """
    def __init__(
            self,
            topic_freq_df,
            top_relevant_terms_df,
            ):

        self.topic_freq_df = topic_freq_df
        self.topic_freq_df['Category'] = (
            'Topic' +
            (self.topic_freq_df['dominant_topic'].astype(int) + 1).astype(str)
        )
        self.topic_freq_df.set_index('Category', drop=False, inplace=True)

        self.top_relevant_terms_df = top_relevant_terms_df
        self.ixs = pd.IndexSlice


    def get_ngram(self, tokenized_sentence_list):
        """
        Get N-grams for nodes & edges.

        Parameters
        ----------

        tokenized_sentence_list: list
            A list of tokenized documents.

        Examples
        --------

        >>> import unipy_nlp.data_collector as udcl
        >>> import unipy_nlp.preprocessing as uprc
        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> import unipy_nlp.network_plot as unet
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.train_lda(...)
        >>> tpm.visualize_lda_to_html(...)
        >>> vnet = unet.WordNetwork(
        ...     topic_freq_df=tpm.topic_freq_df,
        ...     top_relevant_terms_df=tpm.top_relevant_terms_df,
        ... )
        >>> vnet.get_ngram(tokenized)

        """
        ngramed = token_pair_extractor(tokenized_sentence_list)
        ngramed_df = pd.DataFrame(
            ngramed,
            columns=['target', 'context'],
        ).dropna()

        self.ngramed_list = ngramed
        self.ngramed_df = ngramed_df

        return ngramed_df

    def save_ngram(self, filepath, type='json'):
        """
        Save N-grams.

        Parameters
        ----------

        filepath: str
            A filepath to save.

        type: str (default: `'json'`, `{'json', 'csv'}`)
            Choose file type.

        Examples
        --------

        >>> import unipy_nlp.data_collector as udcl
        >>> import unipy_nlp.preprocessing as uprc
        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> import unipy_nlp.network_plot as unet
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.train_lda(...)
        >>> tpm.visualize_lda_to_html(...)
        >>> vnet = unet.WordNetwork(
        ...     topic_freq_df=tpm.topic_freq_df,
        ...     top_relevant_terms_df=tpm.top_relevant_terms_df,
        ... )
        >>> vnet.get_ngram(tokenized)
        >>> vnet.save_ngram('data/_tmp_dump/network_plot/ngram.json', type='json')

        """
        dirpath, filename = os.path.split(filepath)
        os.makedirs(dirpath, exist_ok=True)
        if type == 'json':
            with open(filepath, 'w', encoding='utf-8') as jfile:
                # converted_json = json.dumps(obj)
                json.dump(self.ngramed_list, jfile, ensure_ascii=False)
        elif type == 'csv':
            self.ngramed_df.to_csv(
                filepath,
                index=False,
                header=True,
                encoding='utf-8',
            )
        else:
            raise Exception("'type' should be 'json' or 'csv'.")

    def load_ngram(self, filename, type='json'):
        """
        Load N-grams.

        Parameters
        ----------

        filepath: str
            A filepath to save.

        type: str (default: `'json'`, `{'json', 'csv'}`)
            Choose file type.

        Examples
        --------

        >>> import unipy_nlp.data_collector as udcl
        >>> import unipy_nlp.preprocessing as uprc
        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> import unipy_nlp.network_plot as unet
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.train_lda(...)
        >>> tpm.visualize_lda_to_html(...)
        >>> vnet = unet.WordNetwork(
        ...     topic_freq_df=tpm.topic_freq_df,
        ...     top_relevant_terms_df=tpm.top_relevant_terms_df,
        ... )
        >>> vnet.get_ngram(tokenized)
        >>> vnet.save_ngram('data/_tmp_dump/network_plot/ngram.json', type='json')
        >>> vnet.load_ngram('data/_tmp_dump/network_plot/ngram.json', type='json')

        """
        if type == 'json':
            with open(filename, 'r') as jfile:
                self.ngramed_list = json.load(jfile)
                self.ngramed_df = pd.DataFrame(
                    self.ngramed_list,
                    columns=['target', 'context'],
                ).dropna()
        elif type == 'csv':
            self.ngramed_df = pd.read_csv(
                filename,
                encoding='utf-8',
            )
            self.ngramed_list = list(self.ngramed_df.itertuple(index=False))
        else:
            raise Exception("'type' should be 'json' or 'csv'.")


    def _get_linked(self):
        counted = (
            self.ngramed_df
            .groupby(['target', 'context'])
            [['context']]
            .agg('count')
            .rename(columns={'context': 'freq'})
        ).reset_index()

        linked = (
            counted
            .sort_values(by='freq', ascending=False)
        )
        # linked_pairs = (
        #     linked[
        #         (linked['target'] != linked['context'])
        #     ]
        # )
        # graphed_pairs = (
        #     linked_pairs
        #     .groupby('target')
        #     ['context']
        #     .apply(tuple)
        #     .reset_index()
        # )
        # graphed_pairs = (
        #     graphed_pairs[graphed_pairs['context'].apply(len) == 1]
        # )

        self.linked = linked


    def _set_terminfo(self, top_n=30):

        top_relevant_terms_df = self.top_relevant_terms_df
        lda_freq = self.topic_freq_df
        ixs = self.ixs

        r_terms_df = (
            top_relevant_terms_df
            .loc[
                # top_relevant_terms_df.index.levels[0].drop('Default'),
                ixs[lda_freq['Category'].unique(), :],
                ['Term', 'rank', 'Total']
            ]
            .reset_index('Category')
        )
        top_n_terms_df = (
            r_terms_df
            .groupby(['Category'])
            .apply(lambda x: x.sort_values(['rank']).head(top_n))
        )

        terms_info_df = (
            top_n_terms_df[['Category', 'Term', 'rank']]
            .groupby(['Term'])
            .apply(lambda x: x.loc[x['rank'].idxmin(), ['Category']])
        )
        topic_words_df = (
            top_relevant_terms_df
            .loc[top_relevant_terms_df.index.levels[0].drop('Default'), :]
            .groupby(level=0)
            .apply(lambda x: r', '.join(x['Term'].head(top_n)))
        )

        self.top_n_terms_df = top_n_terms_df
        self.terms_info_df = terms_info_df
        self.topic_words_df = topic_words_df

    def _set_colormap(self, cmap='tab10'):

        terms_info_df = self.terms_info_df
        lda_freq = self.topic_freq_df
        linked = self.linked

        topic_list = lda_freq['Category'].unique().tolist()
        colormap = cm.get_cmap(cmap)
        rgb_list = colormap.colors

        topic_rgb_dict = {
            topic: rgb
            for topic, rgb
            in zip(topic_list, rgb_list)
        }
        topic_hex_dict = {
            topic: mpl.colors.rgb2hex(rgb)
            for topic, rgb
            in zip(topic_list, rgb_list)
        }
        terms_info_df['rgb'] = (
            terms_info_df
            ['Category']
            .apply(
                lambda x: topic_rgb_dict.get(x, (1., 1., 1.))
            )
        )
        terms_info_df['hex'] = (
            terms_info_df
            ['Category']
            .apply(
                lambda x: topic_hex_dict.get(x, '#ffffff')
            )
        )

        # Word & HEX
        terms_id_df = pd.DataFrame(
            pd.concat(
                [
                    linked['target'],
                    linked['context'],
                ]
            )
            .unique(),
            columns=['Term'],
        )
        terms_id_df['id'] = terms_id_df.index

        terms_adv_info_df = pd.merge(
            terms_id_df,
            terms_info_df,
            on=['Term'],
            how='left',
        ).dropna()
        topic_node_df = terms_adv_info_df

        linked_filtered = (
            linked[
                (linked['target'] != linked['context']) &
                (linked['target'].isin(topic_node_df['Term'])) &
                (linked['context'].isin(topic_node_df['Term']))
            ]
        ).dropna()

        linked_joined = pd.merge(
            linked_filtered,
            topic_node_df,
            left_on='target',
            right_on='Term',
            how='inner',
        ).reset_index(drop=True)

        to_merged = pd.merge(
            topic_node_df,
            linked_filtered,
            left_on='Term',
            right_on='target',
            how='inner',
        )

        to_topic_freq = to_merged.groupby(['Category'])['freq'].sum()
        from_merged = pd.merge(
            topic_node_df,
            linked_filtered,
            left_on='Term',
            right_on='context',
            how='inner',
        )
        from_topic_freq = from_merged.groupby(['Category'])['freq'].sum()
        # topic_freq = to_topic_freq + from_topic_freq


        self.terms_info_df = terms_info_df
        self.topic_node_df = topic_node_df
        self.topic_freq_df = lda_freq
        self.linked = linked_joined
        self.from_topic_freq = from_topic_freq


    def _get_term_rgb(self, term):
        terms_info = self.terms_info_df
        try:
            return terms_info.loc[term, ['rgb']][0]
        # If KeyError, It belongs to None.
        except KeyError:
            return (1., 1., 1.)


    def _get_term_hex(self, term):
        terms_info = self.terms_info_df
        try:
            return terms_info.loc[term, ['hex']][0]
        # If KeyError, It belongs to None.
        except KeyError:
            return '#ffffff'


    def _get_topic_term_freq(self, topic, term):
        df = self.top_n_terms_df
        return df.loc[
            (df['Category'] == topic) & (df['Term'] == term),
            'Total'
        ][0]


    def _add_topic_node(
            self,
            term,
            freq,
            term_hex,
            group=None,
            ):

        self.pyvis_net.add_node(
            # n_id=term_id,
            n_id=term,
            label=term,
            share='dot',
            color=term_hex,
            title=term,
            value=freq,
            group=None,
        )

    def _add_udf_node(
            self,
            pyvis_net,
            term,
            freq,
            ):
        tinfo_df = self.topic_node_df
        pyvis_net = self.pyvis_net

        term_id, term_hex = (
            tinfo_df
            .loc[tinfo_df['Term'] == term, ['id', 'hex']]
            .values[0]
        )

        pyvis_net.add_node(
            # n_id=term_id,
            n_id=term,
            label=term,
            share='dot',
            color=term_hex,
            title=term,
        )

    def _hidden_option(
            self,
            v,
            thres=10,
            except_cond=True
            ):

        res = False
        if isinstance(v, float) and (except_cond):
            if v >= thres:
                res = True
            else:
                res = False
        else:
            res = False
        # return True if v >= thres else False
        return res

    def _sum_inflow(self, node_id):
        return sum(
            [
                e['value'] for e in self.pyvis_net.edges
                if e['to'] == node_id
            ]
        )

    def _get_inflow(self, from_node, to_node):
        return int(sum(
            [
                e['value'] for e in self.pyvis_net.edges
                if (e['from'] == from_node) & (e['to'] == to_node)
            ]
        ))


    def draw(
            self,
            height="700px",
            width='800px',  # "100%",
            bgcolor='#ffffff',  # "#222222",
            font_color='black',  # "white",
            directed=True,
            notebook=False,
            topic_top_n=None,
            node_freq_threshold=None,
            show_buttons=True,
            ):
        """
        Draw `pyvis.network.Network` using N-grams.

        Parameters
        ----------
        height: str (default: `"700px"`)
            Height of the network plot. It can be pixel-based or percentage-based.
        
         width: str (default: `"800px"`)
            Height of the network plot. It can be pixel-based or percentage-based.
        
        bgcolor: str (default: `'#ffffff'`)
            HEX color for background.
        
        font_color: str (default: `'black'`)
            HEX color or colorname for font.
        
        directed: bool (default: `True`)
            An option to show direction for each edges.
        
        notebook: bool (default: `False`)
            An option to show in jupyter notebook
        
        topic_top_n: int (default: `None`)
            A topic number to show. It depends on its frequency.
        
        node_freq_threshold: int (default: `None`)
            A threshold number to show nodes.
            It is useful when your nodes & edges are too many to show.
            
        show_buttons: bool (default: `True`)
            An option to show interactive buttons in html.

        Example
        -------

        >>> import unipy_nlp.data_collector as udcl
        >>> import unipy_nlp.preprocessing as uprc
        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> import unipy_nlp.network_plot as unet
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.train_lda(...)
        >>> tpm.visualize_lda_to_html(...)
        >>> vnet = unet.WordNetwork(
        ...     topic_freq_df=tpm.topic_freq_df,
        ...     top_relevant_terms_df=tpm.top_relevant_terms_df,
        ... )
        >>> vnet.get_ngram(tokenized)
        >>> vnet.draw(
        ...     height="100%",
        ...     width='800px',
        ...     bgcolor='#ffffff',
        ...     font_color='black',
        ...     directed=True,
        ...     topic_top_n=5,
        ...     node_freq_threshold=100,
        ...     show_buttons=True,
        ... )

        """
        self._get_linked()
        self._set_terminfo()
        self._set_colormap(cmap='tab10')


        self.pyvis_net = net(
            height=height,
            width=width,  # "100%",
            bgcolor=bgcolor,  # "#222222",
            font_color=font_color,  # "white",
            directed=directed,
            notebook=notebook,
        )

        self.pyvis_net.barnes_hut(
            gravity=-12500,
            central_gravity=12,
            spring_length=120,  # 180,
            spring_strength=.01,  # 0.04,
            damping=0.9,  # 0.25
            overlap=0.1,  # 0.
        )
        topic_freq_dict = self.from_topic_freq.to_dict()
        topic_words_df = self.topic_words_df
        topic_node_df = self.topic_node_df
        linked = self.linked

        if node_freq_threshold is not None:
            linked = linked[
                # linked['freq'] >= max(100, linked['freq'].quantile(.5))
                linked['freq'] >= node_freq_threshold
            ]

        if topic_top_n is not None:
            topic_freq_df = self.topic_freq_df[:topic_top_n]
            topic_node_df = topic_node_df[
                topic_node_df['Category'].isin(topic_freq_df.index)
            ]
            linked = linked[linked['Category'].isin(topic_freq_df.index)]
        # terms_adv_info_df

        for topic_row in topic_node_df.itertuples():
            idx, term, _, topic, rgb_str, hex_str = topic_row
            self.pyvis_net.add_node(
                # n_id=term_id,
                n_id=topic,
                label=topic,
                group=topic,
                level=3,
                # The types with the label inside of it are:
                # ellipse, circle, database, box, text.
                # The ones with the label outside of it are:
                # image, circularImage, diamond, dot, star, triangle, triangleDown, square and icon.
                shape='circle',
                # image=wcloud_path,
                # brokenImage=wcloud_name,
                borderWidth=1,
                mass=5,
                color=hex_str,
                title=topic_words_df[topic],
                size=int(topic_freq_dict[topic] / 100),
            )
            self.pyvis_net.add_node(
                # n_id=term_id,
                n_id=term,
                label=term,
                level=2,
                group=topic,
                shape='circle',  # 'text' if freq < 120 else 'circle',
                color=hex_str,
                title=topic,
                size=10,
                borderWidthSelected=10,
            )
            # add_topic_node(self.pyvis_net, term=topic, group=topic, freq=int(topic_freq[topic] * 10000), term_hex=hex_str)
            # add_topic_node(self.pyvis_net, term=term, group=topic, freq=.1, term_hex=hex_str)
            self.pyvis_net.add_edge(
                source=topic,
                to=term,
                value=self._get_topic_term_freq(topic, term),
                arrowStrikethrough=True,
                borderWidthSelected=10,
            )

        # self.pyvis_net.repulsion()
        for link_row in linked.itertuples():
            (
                idx, target, context, freq,
                _term, _id, _topic,
                _rgb_tuple, hex_str,
            ) = link_row

            try:
                hex_str = '#e3e3e3' if hex_str == '#ffffff' else hex_str
            except TypeError as err:
                raise err

            self.pyvis_net.add_node(
                # n_id=term_id,
                n_id=target,
                label=target,
                level=0,
                group=topic,
                shape='circle',  # 'text' if freq < 120 else 'circle',
                color=hex_str,
                title=target,
                hidden=(
                    self._hidden_option(freq, thres=node_freq_threshold)
                    if node_freq_threshold else None
                ),
                size=freq,
                borderWidthSelected=10,
            )
            self.pyvis_net.add_node(
                # n_id=term_id,
                n_id=context,
                label=context,
                level=1,
                # group=topic,
                shape='circle',  # 'text' if freq < 120 else 'circle',
                color=hex_str,
                title=context,
                hidden=(
                    self._hidden_option(freq, thres=node_freq_threshold)
                    if node_freq_threshold else None
                ),
                size=freq,
                borderWidthSelected=10,
            )
            # add_udf_node(self.pyvis_net, term=context, freq=10)
            # add_udf_node(self.pyvis_net, term=target, freq=10)
            self.pyvis_net.add_edge(
                source=context,
                to=target,
                value=freq,
                arrowStrikethrough=True,
            )

        # ids_per_topic = self.topic_node_df.groupby(['Category'])['id'].unique()
        ids_per_topic = (
            topic_node_df
            .groupby(['Category'])
            ['id']
            .apply(lambda x: list(set(x)))
        )
        # for topic, id_list in ids_per_topic.iteritems():
        #     self.pyvis_net.neighbors(id_list)

        neighbor_map = self.pyvis_net.get_adj_list()
        node_list = self.pyvis_net.nodes
        node_dict = {n['id']: n for n in node_list}
        term_node_list = [n for n in node_list if 'Topic' not in n['id']]
        edge_list = self.pyvis_net.edges
        #neighbors(node)

        intersection_node_list = []
        intersection_node_dict = {}
        for node in term_node_list:
            neighbor_ids = neighbor_map[node['id']]

            neighbor_inflow_list = sorted(
                [
                    (n_id, self._get_inflow(node['id'], n_id), node_dict[n_id]['group'])
                    for n_id in neighbor_ids
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            n_df = pd.DataFrame(neighbor_inflow_list, columns=['id', 'inflow', 'grp'])
            grp_total_df = (
                n_df.groupby(['grp'])[['inflow']].sum()
                .sort_values(['inflow'], ascending=False)
            )
            grp_total_df['weight'] = grp_total_df['inflow'] / \
                grp_total_df['inflow'].sum()
            # grp_std = grp_total_df['weight'].std()
            grp_std = np.std(grp_total_df['inflow'])
            grp_total_inflow_dict = grp_total_df['inflow'].to_dict()

            # if len(grp_total_inflow_dict) > 1:
            if True:
                intersection_node_dict.setdefault(
                    node['id'],
                    {
                        'df': grp_total_df,
                        'std': grp_std,
                        'sum': grp_total_df['inflow'].sum(),
                        'inflow': grp_total_inflow_dict,
                        'topic_num': grp_total_df.shape[0],
                    },
                )
                intersection_node_list += [
                    {
                        'id': node['id'],
                        'df': grp_total_df,
                        'std': grp_std,
                        'sum': grp_total_df['inflow'].sum(),
                        'inflow': grp_total_inflow_dict,
                        'topic_num': grp_total_df.shape[0],
                    }
                ]

            title_str_list = [f"{n}({v}, {g})" for n, v, g in neighbor_inflow_list]
            node['title'] = ', '.join(title_str_list) + \
                r'<br>' + str(grp_total_inflow_dict)
            node['size'] = int(np.clip(
                sum([x[1] for x in neighbor_inflow_list]) / 100,
                node['size'],
                75,
            ))

        for node in node_list:
            node['font'].setdefault('size', 64)
            node['font'].setdefault('style', 'bold')
            node['font'].setdefault('shadow', .2)
            node['font'].setdefault('outline', 1.)
            # node["title"] += r"\nNeighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
            # node["value"] = len(neighbor_map[node["id"]])
            # node['shape'] = 'text' if node['size'] < 1000 else 'circle'

        # self.pyvis_net.set_edge_smooth('dynamic')

        # ‘dynamic’, ‘continuous’, ‘discrete’,
        # ‘diagonalCross’, ‘straightCross’,
        # ‘horizontal’, ‘vertical’, ‘curvedCW’, ‘curvedCCW’, ‘cubicBezier’.
        self.pyvis_net.set_edge_smooth('dynamic')
        if show_buttons:
            self.pyvis_net.show_buttons(filter_=['physics'])

        self.intersection_node_list = intersection_node_list


    def save(self, filepath_html):
        """
        Save `pyvis.network.Network`.

        Parameters
        ----------

        filepath_html: str
            A filepath to save.

        Examples
        --------

        Example
        -------

        >>> import unipy_nlp.data_collector as udcl
        >>> import unipy_nlp.preprocessing as uprc
        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> import unipy_nlp.network_plot as unet
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.train_lda(...)
        >>> tpm.visualize_lda_to_html(...)
        >>> vnet = unet.WordNetwork(
        ...     topic_freq_df=tpm.topic_freq_df,
        ...     top_relevant_terms_df=tpm.top_relevant_terms_df,
        ... )
        >>> vnet.get_ngram(tokenized)
        >>> vnet.draw(
        ...     height="100%",
        ...     width='800px',
        ...     bgcolor='#ffffff',
        ...     font_color='black',
        ...     directed=True,
        ...     topic_top_n=5,
        ...     node_freq_threshold=100,
        ...     show_buttons=True,
        ... )
        >>> vnet.save('data/_tmp_dump/network_plot/vnet.html')

        """
        dirpath, filename = os.path.split(filepath_html)
        os.makedirs(dirpath, exist_ok=True)
        self.pyvis_net.save_graph(filepath_html)
        print(f"Saved: '{filepath_html}'")
    

    def get_topic_mutuality_score_dict(
            self,
            cdict,
            ):
        """
        Get scores of terms, based on its mutuality.

        Parameters
        ----------

        cdict: `gensim.corpora.dictionary.Dictionary`
            A corpus dictionary for given documents.

        Examples
        --------

        Example
        -------

        >>> import unipy_nlp.data_collector as udcl
        >>> import unipy_nlp.preprocessing as uprc
        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> import unipy_nlp.network_plot as unet
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.train_lda(...)
        >>> tpm.visualize_lda_to_html(...)
        >>> vnet = unet.WordNetwork(
        ...     topic_freq_df=tpm.topic_freq_df,
        ...     top_relevant_terms_df=tpm.top_relevant_terms_df,
        ... )
        >>> vnet.get_ngram(tokenized)
        >>> vnet.draw(
        ...     height="100%",
        ...     width='800px',
        ...     bgcolor='#ffffff',
        ...     font_color='black',
        ...     directed=True,
        ...     topic_top_n=5,
        ...     node_freq_threshold=100,
        ...     show_buttons=True,
        ... )
        >>> vnet.save('data/_tmp_dump/network_plot/vnet.html')
        >>> (score_dict,
        ... score_dict_indiced) = vnet.get_topic_mutuality_score_dict(
        ...     cdict=tpm.corpora_dict
        ... )

        """

        inode_list = self.intersection_node_list
        volume_score = None
        connectivity_score = None
        equality_score = None

        # v_score_list = [n['sum'] / max_volume for n in inode_list]
        # c_score_list = [n['topic_num'] / max_topic_num for n in inode_list]
        # e_score_list = [n['std'] / max_equality for n in inode_list]
        v_score_list = [n['sum'] for n in inode_list]
        c_score_list = [n['topic_num'] for n in inode_list]
        e_score_list = [n['std'] for n in inode_list]

        scaled_v = minmax_scale(v_score_list)
        scaled_c = minmax_scale(c_score_list)
        scaled_e = minmax_scale(e_score_list)

        zipped = zip(inode_list, scaled_v, scaled_c, scaled_e)

        i_score_dict = {
            n['id']: np.mean([v, c, e]) for n, v, c, e in zipped
        }
        i_score_dict_indiced = {
            cdict.token2id[k]: v for k, v in i_score_dict.items()}

        self.i_score_dict = i_score_dict
        self.i_score_dict_indiced = i_score_dict_indiced

        return i_score_dict, i_score_dict_indiced


    def _get_bow_score(self, bow, score_dict):
        return sum([score_dict.get(i, 0) * min(cnt, 3) for i, cnt in bow])

    def get_network_scored_repr_docs(
            self,
            bow_corpus,
            repr_docs,
            save_ok=True,
            filepath=None,
            ):
        """
        Get representitive documents, based on the mutuality score of terms.

        Parameters
        ----------

        bow_corpus: list
            A nested list, which contains converted documents into a list of token words.

        repr_docs: list
            A list of raw documents.
        
        save_ok: bool (default: `True`)
            An option to save.
        
        filepath: str (default: `None`)
            A filepath to save.

        Examples
        --------

        Example
        -------

        >>> import unipy_nlp.data_collector as udcl
        >>> import unipy_nlp.preprocessing as uprc
        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> import unipy_nlp.network_plot as unet
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.train_lda(...)
        >>> tpm.visualize_lda_to_html(...)
        >>> vnet = unet.WordNetwork(
        ...     topic_freq_df=tpm.topic_freq_df,
        ...     top_relevant_terms_df=tpm.top_relevant_terms_df,
        ... )
        >>> vnet.get_ngram(tokenized)
        >>> vnet.draw(
        ...     height="100%",
        ...     width='800px',
        ...     bgcolor='#ffffff',
        ...     font_color='black',
        ...     directed=True,
        ...     topic_top_n=5,
        ...     node_freq_threshold=100,
        ...     show_buttons=True,
        ... )
        >>> vnet.save('data/_tmp_dump/network_plot/vnet.html')
        >>> (score_dict,
        ... score_dict_indiced) = vnet.get_topic_mutuality_score_dict(
        ...     cdict=tpm.corpora_dict
        ... )
        >>> core_repr = vnet.get_network_scored_repr_docs(
        ...     bow_corpus=repr_bow_corpus_doc,
        ...     repr_docs=repr_sentenced,
        ...     save_ok=True,
        ...     savepath=None,
        ... )

        """
        scored = sorted(
            [
                (i, self._get_bow_score(bow, self.i_score_dict_indiced))
                for i, bow in enumerate(bow_corpus)
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        scored_s_only = [s for i, s in scored]

        plt.hist(scored_s_only)
        plt.title('Count by Score')
        plt.xlabel('Score')
        plt.ylabel('Count')

        scored_repr = [
            (i, repr_idx, score, repr_docs[repr_idx])
            for i, (repr_idx, score) in enumerate(scored)
        ]
        scored_repr_df = pd.DataFrame(
            scored_repr,
            columns=['rank', 'repr_idx', 'score', 'text'],
        )

        self.scored_repr_df = scored_repr_df

        if save_ok:
            if filepath is None:
                raise ValueError("'filepath' should be given when 'save_ok' is `True`.")
            else:
                scored_repr_df.to_csv(
                    filepath,
                    index=False,
                    header=True,
                    encoding='utf-8',
                )

        return scored_repr_df


