# -*- coding: utf-8 -*-
"""
Created on Tue Jun 01 15:41:19 2019

@author: Youngju Jaden Kim
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


# {nouned, morphed, morphed_filtered, spmed, spmed_unspaced}
TOKENIZED_NM = 'spmed_unspaced'

with open(f'data/{TOKENIZED_NM}.json', 'r') as jfile:
    tokenized = json.load(jfile)
    print(len(tokenized))

cdict = corpora_dict(tokenized)
# %%

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


#%%
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


# %% Load NGRAM ---------------------------------------------------------------------

TOKENIZED_NM = 'spmed_unspaced'
NGRAM_SAVED_OK = True

ngramed_filename = f'data/{TOKENIZED_NM}_skipgram_4window_8grams'
if not NGRAM_SAVED_OK:
    ngramed = token_pair_extractor(tokenized)
    ngramed_df = pd.DataFrame(
        ngramed,
        columns=['target', 'context'],
    ).dropna()

    with open(f'{ngramed_filename}.json', 'w', encoding='utf-8') as jfile:
        # converted_json = json.dumps(obj)
        json.dump(ngramed, jfile, ensure_ascii=False)

    ngramed_df.to_csv(
        f'{ngramed_filename}.csv',
        index=False,
        header=True,
        encoding='utf-8',
    )

else:
    with open(f'{ngramed_filename}.json', 'r') as jfile:
        ngramed = json.load(jfile)
        ngramed_df = pd.DataFrame(
            ngramed,
            columns=['target', 'context'],
        )
        print(ngramed_df.shape)


counted = (
    ngramed_df
    .groupby(['target', 'context'])
    [['context']]
    .agg('count')
    .rename(columns={'context': 'freq'})
).reset_index()

linked = (
    counted
    .sort_values(by='freq', ascending=False)
)

linked.to_csv(
    'data/linked.csv',
    index=False,
    header=True,
    encoding='utf-8',
)


linked_pairs = (
    linked[
        (linked['target'] != linked['context'])
    ]
)
graphed_pairs = (
    linked_pairs
    .groupby('target')
    ['context']
    .apply(tuple)
    .reset_index()
)
graphed_pairs = graphed_pairs[graphed_pairs['context'].apply(len) == 1]
print(graphed_pairs.shape)
graphed_pairs.head()



# %% Top Relevent Terms

LDA_TOPIC_NUM = 5

lda_freq = pd.read_csv(
    f'lda_frequencies_{LDA_TOPIC_NUM}topic.csv',
).sort_values(['doc_num'], ascending=False).head(5)
lda_freq.loc[:, 'Category'] = 'Topic' + \
    (lda_freq['dominant_topic'] + 1).astype(str)
lda_freq.set_index('Category', drop=False, inplace=True)

top_relevant_terms_df = pd.read_csv(
    f'data/{LDA_TOPIC_NUM}topic_top_relevant_terms_df.csv',
    index_col=[0, 1],
)
top_relevant_terms_df.head(20)


TOP_N_TERM = 30

ixs = pd.IndexSlice

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
    .apply(lambda x: x.sort_values(['rank']).head(TOP_N_TERM))
)

terms_info_df = (
    top_n_terms_df[['Category', 'Term', 'rank']]
    .groupby(['Term'])
    .apply(lambda x: x.loc[x['rank'].idxmin(), ['Category']])
)

# Set a colormap
# topic_list = terms_info_df['Category'].unique()
topic_list = lda_freq['Category'].unique().tolist()
colormap = cm.get_cmap('tab10')
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
terms_info_df.head()


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
topic_node_df.head()

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

linked_final = linked_joined[
    linked_final['freq'] >= max(100, linked_final['freq'].quantile(.5))
]

# %% LOAD TOP DOCS
TOP_DOC_NUM = 5

top_docs_df = pd.read_csv(
    f'data/dominant_{LDA_TOPIC_NUM}topic_kwd_sorted_top{TOP_DOC_NUM}_df.csv'
)
topic_words_df = (
    top_relevant_terms_df
    .loc[top_relevant_terms_df.index.levels[0].drop('Default'), :]
    .groupby(level=0)
    .apply(lambda x: r', '.join(x['Term'].head(TOP_N_TERM)))
)
topic_joined = topic_node_df.groupby(
    ['Category'])['Term'].apply(lambda x: ', '.join(x))


# %% Plot

def get_term_rgb(term, terms_info=terms_info_df):
    try:
        return terms_info.loc[term, ['rgb']][0]
    # If KeyError, It belongs to None.
    except KeyError as err:
        return (1., 1., 1.)


def get_term_hex(term, terms_info=terms_info_df):
    try:
        return terms_info.loc[term, ['hex']][0]
    # If KeyError, It belongs to None.
    except KeyError as err:
        return '#ffffff'


def get_topic_term_freq(topic, term, df=top_n_terms_df):
    return df.loc[
        (df['Category'] == topic) & (df['Term'] == term),
        'Total'
    ][0]


def add_topic_node(
        pyvis_net,
        term,
        freq,
        term_hex,
        group=None,
):

    pyvis_net.add_node(
        # n_id=term_id,
        n_id=term,
        label=term,
        share='dot',
        color=term_hex,
        title=term,
        value=freq,
        group=None,
    )


def add_udf_node(
        pyvis_net,
        term,
        freq,
        tinfo_df=terms_adv_info_df,
        linked_df=linked_filtered,
):

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


def hiden_option(v, thres=10, except_cond=True):

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


def sum_inflow(node_id):
    return sum([e['value'] for e in voice_net.edges if e['to'] == node_id])


def get_inflow(from_node, to_node):
    return int(sum([e['value'] for e in voice_net.edges
                    if (e['from'] == from_node) & (e['to'] == to_node)]))


# %%

NODE_THRESHOLD = 200

voice_net = net(
    height="700px",
    width='800px',  # "100%",
    bgcolor='#ffffff',  # "#222222",
    font_color='black',  # "white",
    directed=True,
    notebook=True,
)
voice_net.barnes_hut(
    gravity=-12500,
    central_gravity=12,
    spring_length=100,  # 180,
    spring_strength=.01,  # 0.04,
    damping=0.25,
    overlap=0.,
)


# topic_freq_dict = lda_freq['doc_num'].to_dict()
topic_freq_dict = from_topic_freq.to_dict()

for topic_row in topic_node_df.itertuples():
    idx, term, _, topic, rgb_str, hex_str = topic_row
# for topic_row in topic_node_df.dropna().itertuples():
#     idx, topic, term, hex_str = topic_row

#     wcloud = WordCloud(
#         font_path=font_path,
#         background_color=None,
#         # background_color='rgba(255, 255, 255, 0)',
#         colormap='tab10',
#         mode='RGBA',
#     )
#     topic_term_str = (
#         topic_node_df
#         .groupby(['Category'])
#         ['Term']
#         .apply(lambda x: ', '.join(x))
#     )
#     wcloud.generate(topic_term_str)
#     wcloud_path = f'{os.getcwd()}/data/{topic}_wcloud.png'
#     wcloud.to_file(wcloud_path)

    voice_net.add_node(
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
    voice_net.add_node(
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
    # add_topic_node(voice_net, term=topic, group=topic, freq=int(topic_freq[topic] * 10000), term_hex=hex_str)
    # add_topic_node(voice_net, term=term, group=topic, freq=.1, term_hex=hex_str)
    voice_net.add_edge(
        source=topic,
        to=term,
        value=get_topic_term_freq(topic, term),
        arrowStrikethrough=True,
        borderWidthSelected=10,
    )

# voice_net.repulsion()
for link_row in linked_final.itertuples():
    (
        idx, target, context, freq,
        _term, _id, _topic,
        _rgb_tuple, hex_str,
    ) = link_row

    try:
        hex_str = '#e3e3e3' if hex_str == '#ffffff' else hex_str
    except TypeError as err:
        raise err

    voice_net.add_node(
        # n_id=term_id,
        n_id=target,
        label=target,
        level=0,
        group=topic,
        shape='circle',  # 'text' if freq < 120 else 'circle',
        color=hex_str,
        title=target,
        # hidden=hiden_option(freq, thres=NODE_THRESHOLD),
        size=freq,
        borderWidthSelected=10,
    )
    voice_net.add_node(
        # n_id=term_id,
        n_id=context,
        label=context,
        level=1,
        # group=topic,
        shape='circle',  # 'text' if freq < 120 else 'circle',
        color=hex_str,
        title=context,
        # hidden=hiden_option(freq, thres=NODE_THRESHOLD),
        size=freq,
        borderWidthSelected=10,
    )
    # add_udf_node(voice_net, term=context, freq=10)
    # add_udf_node(voice_net, term=target, freq=10)
    voice_net.add_edge(
        source=context,
        to=target,
        value=freq,
        arrowStrikethrough=True,
    )

# ids_per_topic = terms_adv_info_df.groupby(['Category'])['id'].unique()
ids_per_topic = (
    terms_adv_info_df
    .groupby(['Category'])
    ['id']
    .apply(lambda x: list(set(x)))
)
# for topic, id_list in ids_per_topic.iteritems():
#     voice_net.neighbors(id_list)

neighbor_map = voice_net.get_adj_list()
node_list = voice_net.nodes
node_dict = {n['id']: n for n in node_list}
term_node_list = [n for n in node_list if 'Topic' not in n['id']]
edge_list = voice_net.edges
#neighbors(node)

intersection_node_list = []
intersection_node_dict = {}
for node in term_node_list:
    neighbor_ids = neighbor_map[node['id']]

    neighbor_inflow_list = sorted(
        [
            (n_id, get_inflow(node['id'], n_id), node_dict[n_id]['group'])
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

# voice_net.set_edge_smooth('dynamic')

# ‘dynamic’, ‘continuous’, ‘discrete’,
# ‘diagonalCross’, ‘straightCross’,
# ‘horizontal’, ‘vertical’, ‘curvedCW’, ‘curvedCCW’, ‘cubicBezier’.
voice_net.set_edge_smooth('dynamic')
voice_net.show_buttons(filter_=['physics'])
# nxg = G# nx.complete_graph(30)
# got_net.from_nx(nxg)
voice_net.save_graph(f'data/{TOKENIZED_NM}_voice_net.html')
print(f"Saved: 'data/{TOKENIZED_NM}_voice_net.html'")
# voice_net.show(f'data/{TOKENIZED_NM}_voice_net.html')


# %% Scoring

def get_topic_mutuality_score_dict(
        inode_list,
):

    volume_score = None
    connectivity_score = None
    equality_score = None

#     v_score_list = [n['sum'] / max_volume for n in inode_list]
#     c_score_list = [n['topic_num'] / max_topic_num for n in inode_list]
#     e_score_list = [n['std'] / max_equality for n in inode_list]
    v_score_list = [n['sum'] for n in inode_list]
    c_score_list = [n['topic_num'] for n in inode_list]
    e_score_list = [n['std'] for n in inode_list]

    scaled_v = minmax_scale(v_score_list)
    scaled_c = minmax_scale(c_score_list)
    scaled_e = minmax_scale(e_score_list)

    zipped = zip(inode_list, scaled_v, scaled_c, scaled_e)

    return {n['id']: np.mean([v, c, e]) for n, v, c, e in zipped}


i_score_dict = get_topic_mutuality_score_dict(intersection_node_list)
i_score_dict_indiced = {cdict.token2id[k]: v for k, v in i_score_dict.items()}


def get_topic_mutuality_hr_score_dict(
        inode_list,
):

    volume_score = None
    connectivity_score = None
    equality_score = None

    v_score_list = [n['sum'] for n in inode_list]
    c_score_list = [n['topic_num'] for n in inode_list]
    e_score_list = [n['std'] for n in inode_list]

    scaled_v = minmax_scale(v_score_list)
    scaled_c = minmax_scale(c_score_list)
    scaled_e = minmax_scale(e_score_list)

    scaled_d = (scaled_c + scaled_e) / 2.

    zipped = zip(inode_list, scaled_v, scaled_d)
    # zipped = zip(inode_list, v_score_list, d_score_list)

    return {n['id']: {'frequency': v, 'influence': d} for n, v, d in zipped}


j_score_dict = get_topic_mutuality_hr_score_dict(intersection_node_list)
j_score_dict_indiced = {cdict.token2id[k]: v for k, v in j_score_dict.items()}


def get_bow_score(bow, score_dict):
    return sum([score_dict.get(i, 0) * min(cnt, 3) for i, cnt in bow])

scored = sorted(
    [
        (i, get_bow_score(bow, i_score_dict_indiced))
        for i, bow in enumerate(bow_corpus_raw_repr)
    ],
    key=lambda x: x[1],
    reverse=True,
)
scored_s_only = [s for i, s in scored]
plt.hist(scored_s_only)


# %% Root-Cause Representation

scored_repr = [
    (i, repr_idx, score, repr_sentenced[repr_idx])
    for i, (repr_idx, score) in enumerate(scored)
]
scored_repr_df = pd.DataFrame(
    scored_repr,
    columns=['rank', 'repr_idx', 'score', 'text'],
)
scored_repr_df.to_csv(
    f'data/scored_repr_{LDA_TOPIC_NUM}_topics.csv',
    index=False,
    header=True,
    encoding='utf-8',
)
# pd.set_option('display.max_colwidth', 300)

print(f'TOPIC N: {LDA_TOPIC_NUM}')
scored_repr_df.head(30)
