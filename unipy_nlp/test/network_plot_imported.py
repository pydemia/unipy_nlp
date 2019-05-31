#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../happymap/scripts'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Network Plot

#%%
import os
import sys
import json
import itertools as it
import warnings
import numpy as np
import pandas as pd
from pprint import pprint
from wordcloud import WordCloud

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm

import gensim
from gensim.corpora import Dictionary as corpora_dict
import sentencepiece as spm
from konlpy.tag import Kkma, Mecab, Okt

# pd.options.mode.chained_assignment = None
fpath = '../data'

# warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['font.serif'] = 'NanumMyeongjo'
mpl.rcParams['font.sans-serif'] = 'NanumGothic'
mpl.rcParams['font.monospace'] = 'NanumGothicCoding'

[f.name for f in fm.fontManager.ttflist if 'nanum' in f.name.lower()]

font_dict = {
    path.split('/')[-1][:-4]: path
    for path in fm.get_fontconfig_fonts()
    if 'nanum' in path.lower().split('/')[-1]
}
font_path = font_dict['NanumBarunGothic']

#%% [markdown]
# ## Data Preprocessing

#%%
TOKENIZED_NM = 'spmed_unspaced'  # {nouned, morphed, morphed_filtered, spmed, spmed_unspaced}

with open(f'data/{TOKENIZED_NM}.json', 'r') as jfile:
    tokenized = json.load(jfile)
    print(len(tokenized))

cdict = corpora_dict(tokenized)

tagger = Mecab()


new_file = f'data/full_sentence.txt'
with open(new_file, 'r') as file:
    sentenced = file.readlines(file)

tag_list = [
    '체언 접두사', '명사', '한자', '외국어',
    '수사', '구분자',
    '동사',
    '부정 지정사', '긍정 지정사',
]
tagset_wanted = [
    tag
    for tag, desc in tagger.tagset.items()
    for key in tag_list
    if key in desc
]
def get_wanted_morphs(s, wanted_tags):
    res_pos = tagger.pos(s)
    
    res = list(
        filter(
            lambda x: (x[1] in wanted_tags) and (len(x[0]) > 1),
            res_pos,
        )
    )
    return [morph[0] for morph in res]

morphed_filtered = [
    get_wanted_morphs(s, tagset_wanted)
    for s in sentenced
]
spm_source = (
    # sentenced
    [' '.join(s) for s in morphed_filtered]
    # [''.join(s) for s in morphed]
)
spm_source_joined = '\n'.join(
    spm_source
)

SPM_VOCAB_SIZE = 50000
SPM_MODEL_TYPE = 'word'  # {unigram (default), bpe, char, word}
SPM_MODEL_NAME = f'happy_spm_{SPM_MODEL_TYPE}_{SPM_VOCAB_SIZE}'


sp = spm.SentencePieceProcessor()
sp.Load(f'{SPM_MODEL_NAME}.model')

spmed = [
    sp.EncodeAsPieces(l) for l in spm_source
]
spmed_ids = [
    sp.EncodeAsIds(l) for l in spm_source
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
#%%
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
#     paired_arr = np.array(
#         skip_gram_pairs(
#             line_token_ids,
#             window_size=window_size,
#             num_grams=num_grams,
#             # negative_samples=1.,
#             shuffle=False,
#         ),
#         # dtype=np.int64,
#         dtype=np.str,
#     )
    # arr_shape = paired_arr.shape
    # print(paired_arr[:, 0])

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
    # token_ids = [line.split() for line in token_ls]
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
#   [
#        writer.write(example.SerializeToString())
#        for example in tf_example_list_gen
#   ]
    # res = sum(list(tf_example_list), [])
    # res = np.stack(tf_example_list_gen)
    res = sum(example_array_list, [])
    return res

#%% [markdown]
# ### Load the Model

#%%
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


#%%
ngramed_df[:5]


#%%
# representitive_sentences_df = pd.read_csv(
#     'data/representitive_sentences_df.csv',
# )
# repr_sentenced = representitive_sentences_df['documents'].tolist()

representitive_short_sentences_df = pd.read_csv(
    'data/representitive_short_sentences_df.csv',
)
repr_sentenced = representitive_short_sentences_df['documents'].tolist()
repr_sentenced = [r for r in repr_sentenced if 10 <= len(r) < 25]


#%%
tagger = Mecab()

tag_list = [
    '체언 접두사', '명사', '한자', '외국어',
    '수사', '구분자',
    '동사',
    '부정 지정사', '긍정 지정사',
]
tagset_wanted = [
    tag
    for tag, desc in tagger.tagset.items()
    for key in tag_list
    if key in desc
]
def get_wanted_morphs(s, wanted_tags):
    res_pos = tagger.pos(s)

    res = list(
        filter(
            lambda x: (x[1] in wanted_tags) and (len(x[0]) > 1),
            res_pos,
        )
    )
    return [morph[0] for morph in res]

morphed_filtered = [
    get_wanted_morphs(s, tagset_wanted)
    for s in repr_sentenced
]
spm_source = (
    # sentenced
    [' '.join(s) for s in morphed_filtered]
    # [''.join(s) for s in morphed]
)
spm_source_joined = '\n'.join(
    spm_source
)

SPM_VOCAB_SIZE = 50000
SPM_MODEL_TYPE = 'word'  # {unigram (default), bpe, char, word}
SPM_MODEL_NAME = f'happy_spm_{SPM_MODEL_TYPE}_{SPM_VOCAB_SIZE}'


sp = spm.SentencePieceProcessor()
sp.Load(f'{SPM_MODEL_NAME}.model')

spmed = [
    sp.EncodeAsPieces(l) for l in spm_source
]
spmed_ids = [
    sp.EncodeAsIds(l) for l in spm_source
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

tokenized_repr = spmed_unspaced
cdict = corpora_dict(tokenized_repr)

bow_corpus_idx_repr = [cdict.doc2idx(doc) for doc in tokenized_repr]
bow_corpus_raw_repr = [cdict.doc2bow(doc) for doc in tokenized_repr]

bow_corpus = bow_corpus_raw_repr


#%%
counted = (
    ngramed_df
    .groupby(['target', 'context'])
    [['context']]
    .agg('count')
    .rename(columns={'context':'freq'})
).reset_index()

linked = (
    counted
    .sort_values(by='freq', ascending=False)
)
linked.head(5)


#%%
linked.to_csv(
    'data/linked.csv',
    index=False,
    header=True,
    encoding='utf-8',
)


#%%
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

graphed = linked_filtered.groupby('target')['context'].apply(tuple).reset_index()
print(graphed.shape)
graphed.tail(5)noded = linked_filtered[['target', 'context']].values.tolist()
# noded = graphed.values.tolist()#%% [markdown]
# ## Top Relevant Terms
#%% [markdown]
# ### Top Topic Term

#%%
LDA_TOPIC_NUM = 5


#%%
lda_freq = pd.read_csv(
    f'lda_frequencies_{LDA_TOPIC_NUM}topic.csv',
).sort_values(['doc_num'], ascending=False).head(5)
lda_freq.loc[:, 'Category'] = 'Topic' + (lda_freq['dominant_topic'] + 1).astype(str)
lda_freq.set_index('Category', drop=False, inplace=True)

top_relevant_terms_df = pd.read_csv(
    f'data/{LDA_TOPIC_NUM}topic_top_relevant_terms_df.csv',
    index_col=[0, 1],
)
top_relevant_terms_df.head(20)


#%%
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


#%%
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


#%%
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

# linked_final.iloc[np.r_[:5, -5:]]


#%%
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

topic_freq = to_topic_freq + from_topic_freq
topic_freq.head()

#%% [markdown]
# ## Plotting
#%% [markdown]
# ### pyvis

#%%
from pyvis.network import Network as net


#%%
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


#%%
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


#%%
TOP_DOC_NUM = 5

top_docs_df = pd.read_csv(
    f'data/dominant_{LDA_TOPIC_NUM}topic_kwd_sorted_top{TOP_DOC_NUM}_df.csv'
)


#%%
topic_words_df = (
    top_relevant_terms_df
    .loc[top_relevant_terms_df.index.levels[0].drop('Default'), :]
    .groupby(level=0)
    .apply(lambda x: r', '.join(x['Term'].head(TOP_N_TERM)))
)

top_relevant_terms_df.groupby(level=0)[['Term']].sort_values('Freq', ascending=False).sum()
#%%
topic_node_df.groupby(['Category'])['Term'].apply(lambda x: ', '.join(x))


#%%
linked_final.shape


#%%
linked_final = linked_joined


#%%
pd.concat([linked_final['target'], linked_final['context']]).unique().shape


#%%
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
        shape='circle', # 'text' if freq < 120 else 'circle',
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
        shape='circle', # 'text' if freq < 120 else 'circle',
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
        shape='circle', # 'text' if freq < 120 else 'circle',
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
    grp_total_df['weight'] = grp_total_df['inflow'] / grp_total_df['inflow'].sum()
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
    node['title'] = ', '.join(title_str_list) + r'<br>' + str(grp_total_inflow_dict)
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

#%% [markdown]
# ## Edited
linked_final.shapelinked_final_subset = linked_final[:100]NODE_THRESHOLD = 200

voice_net = net(
    height="1200px",
    width='1200px',  # "100%",
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
        shape='dot',
        # image=wcloud_path,
        # brokenImage=wcloud_name,
        borderWidth=1,
        mass=5,
        color=hex_str,
        title=topic_words_df[topic],
        size=int(topic_freq_dict[topic] / 250),
    )
    voice_net.add_node(
        # n_id=term_id,
        n_id=term,
        label=term,
        level=2,
        group=topic,
        shape='dot',
        color=hex_str,
        title=topic,
        size=20,
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

erased_list = ['위원', '사람', '팀원', '모르', '아니', 'dt']
# voice_net.repulsion()
for link_row in linked_final_subset.itertuples():
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
        shape='dot',
        color=hex_str,
        title=target,
        hidden=hiden_option(freq, thres=NODE_THRESHOLD, except_cond=(target not in erased_list)),
        size=freq * 25,
        borderWidthSelected=10,
    )
    voice_net.add_node(
        # n_id=term_id,
        n_id=context,
        label=context,
        level=1,
        # group=topic,
        shape='dot',
        color=hex_str,
        title=context,
        hidden=hiden_option(freq, thres=NODE_THRESHOLD, except_cond=(target not in erased_list)),
        size=freq * 25,
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
    grp_total_df['weight'] = grp_total_df['inflow'] / grp_total_df['inflow'].sum()
    # grp_std = grp_total_df['weight'].std()
    grp_std = np.std(grp_total_df['inflow'])
    grp_total_inflow_dict = grp_total_df['inflow'].to_dict()

    if len(grp_total_inflow_dict) > 1:
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
    node['title'] = ', '.join(title_str_list) + r'<br>' + str(grp_total_inflow_dict)
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
    node['label'] = None
    # node["title"] += r"\nNeighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
    # node["value"] = len(neighbor_map[node["id"]])

# voice_net.set_edge_smooth('dynamic')

# ‘dynamic’, ‘continuous’, ‘discrete’,
# ‘diagonalCross’, ‘straightCross’,
# ‘horizontal’, ‘vertical’, ‘curvedCW’, ‘curvedCCW’, ‘cubicBezier’.
voice_net.set_edge_smooth('dynamic')
voice_net.show_buttons(filter_=['physics'])
# nxg = G# nx.complete_graph(30)
# got_net.from_nx(nxg)
voice_net.save_graph(f'data/{TOKENIZED_NM}_voice_net_tmp.html')
print(f"Saved: 'data/{TOKENIZED_NM}_voice_net_tmp.html'")
# voice_net.show(f'data/{TOKENIZED_NM}_voice_net.html')#%% [markdown]
# ### Node Scoring

#%%
from sklearn.preprocessing import minmax_scale


#%%
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


#%%
i_score_dict = get_topic_mutuality_score_dict(intersection_node_list)

i_score_dict_indiced = {cdict.token2id[k]: v for k, v in i_score_dict.items()}


#%%
list(i_score_dict_indiced.items())[0]


#%%
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


#%%
j_score_dict = get_topic_mutuality_hr_score_dict(intersection_node_list)

j_score_dict_indiced = {cdict.token2id[k]: v for k, v in j_score_dict.items()}


#%%
j_score_df = pd.DataFrame(j_score_dict).T
j_score_df.to_csv(
    'data/j_score_df.csv',
    index=True,
    header=True,
    encoding='utf-8',
)


#%%
print(j_score_df.shape)
print(j_score_df.head())


#%%
len(i_score_dict)


#%%
def get_bow_score(bow, score_dict):
    return sum([score_dict.get(i, 0) * min(cnt, 3) for i, cnt in bow])


#%%
scored = sorted(
    [
        (i, get_bow_score(bow, i_score_dict_indiced))
        for i, bow in enumerate(bow_corpus_raw_repr)
    ],
    key=lambda x: x[1],
    reverse=True,
)


#%%
scored_s_only = [s for i, s in scored]

plt.hist(scored_s_only)

#%% [markdown]
# #### Root-Cause Representitation

#%%
len(scored)

morphed_repr = [
    tagger.morphs(s)
    for s in scored_repr
]
#%%
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


#%%
pd.set_option('display.max_colwidth', 300)

print(f'TOPIC N: {LDA_TOPIC_NUM}')
scored_repr_df.head(30)


#%%
rc_dict = corpora_dict(scored_repr_df.head(10)['text'].tolist())


#%%
[scored_repr_df.head(10)['text'].tolist()]


#%%
max(scored)


#%%
i_score_df = pd.DataFrame(list(i_score_dict_indiced.items()), columns=['idx', 'score'])
i_score_df

#%% [markdown]
# #### Topic Relevance Mapping

#%%
from gensim import corpora, models
from gensim.models import CoherenceModel


#%%
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
                decay=.8, # {.5, 1.}
                per_word_topics=False,
                offset=1.,
                iterations=30,
                gamma_threshold=.001, # 0.001,
                minimum_probability=.05,  # .01,
                minimum_phi_value=.01,
                random_state=1,
            )
            coherence_model = CoherenceModel(
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
            coherence_model = CoherenceModel(
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
            urllib.urlretrieve (mallet_url, mallet_filename)

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
            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            # coherence_list += [coherence_model.get_coherence()]

    return model_list, coherence_list


#%%
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


#%%
MODEL_SAVED_OK = True

if not MODEL_SAVED_OK:

    lda_model, model_list, model_dict, coherence_values = pick_best_n_topics(
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


#%%
lda_model = model_dict[7] # 15 is the best
# lda_model = model_list[2] # 15 is the best

LDA_TOPIC_NUM = lda_model.num_topics
LDA_MODEL_NAME = f'happy_lda_{LDA_TOPIC_NUM}topic'


#%%
def get_saliency(tinfo_df):
    """Calculate Saliency for terms within a topic.
    
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


def get_terminfo_table(
        lda_model,
        corpus: list=None,
        dictionary: gensim.corpora.dictionary.Dictionary=None,
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

#         corpus_dict_df = pd.DataFrame(
#             # It is possible
#             # because the keys of this dictionary generated from range(int).
#             # Usually the dictionary is iterable but not ordered.
#             list(dictionary.values()),
#             # [dictionary[i] for i, _ in enumerate(dictionary)],
#             columns=['Term'],
#         )
#         corpus_dict_df['term_id'] = corpus_dict_df.index
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
        topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()  # doc_lengths @ doc_topic_dists
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
        kernel = (topic_given_term * np.log((topic_given_term.T / topic_proportion).T))
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


#%%
(total_terms_df, top_relevant_terms_df,
 r_adj_score_df, bow_score_list) = get_terminfo_table(
    lda_model,
    corpus=bow_corpus,
    dictionary=cdict,
    doc_topic_dists=None,
    use_gensim_prepared=True,
    top_n=30,
)

#%% [markdown]
# ### networkx

#%%
import networkx as nx

noded[:2]top_edges = linked_filtered[:10][['target', 'context']].values.tolist()G = nx.DiGraph(directed=True)
G.add_edges_from(noded)

val_map = {
    '결정': 1.0,
    '의사': 0.5714285714285714,
    'sk': 0.7,
}
node_values = [val_map.get(node, 0.3) for node in G.nodes()]
red_edges = top_edges
edge_colours = [
    'black'
    if not edge in red_edges
    else 'red'
    for edge in G.edges()
]
black_edges = [
    edge
    for edge in G.edges()
    if edge not in red_edges
]

pos = nx.spring_layout(G)

fig = plt.figure(figsize=(18, 18))
nx.draw_networkx_nodes(
    G,
    pos,
    cmap=plt.get_cmap('jet'),
    node_color=node_values,
    node_size=500,
)
nx.draw_networkx_labels(
    G,
    pos,
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=red_edges,
    edge_color='r',
    arrows=True,
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=black_edges,
    # edge_color='r',
    arrows=False,
)

# plt.axis('equal')
# plt.savefig("plot.png", dpi=1000)
plt.show()#%% [markdown]
# ## ego_graph

#%%
# Create a BA model graph
n = 1000
m = 2
G = nx.generators.barabasi_albert_graph(n, m)
# find node with largest degree
node_and_degree = G.degree()
(largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
# Create ego graph of main hub
hub_ego = nx.ego_graph(G, largest_hub)
# Draw graph
pos = nx.spring_layout(hub_ego)
nx.draw(hub_ego, pos, node_color='b', node_size=50, with_labels=False)
# Draw ego as large and red
nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], node_size=300, node_color='r')
plt.show()


#%%
type(G)


#%%
get_ipython().run_line_magic('pinfo', 'G')

#%% [markdown]
# ### Tutorial

#%%
G = nx.Graph()


#%%
G.add_path([0, 1, 2])


#%%
len(G)


#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# create some test graph
graph = nx.erdos_renyi_graph(1000, 0.005)

# create an ego-graph for some node
node = 0
ego_graph = nx.ego_graph(graph, node, radius=2)

# plot to check
nx.draw(ego_graph); plt.show()


#%%
G = nx.Graph()
G.add_edges_from([('a','b'),('a','c'),('b','d'),('b','e'),
                  ('e','h'),('c','f'),('c','g')])


#%%
list(nx.bfs_successors(G, 'b'))


#%%
ego = nx.ego_graph(G, n='b', center=True)


#%%
nx.draw(ego)

#%% [markdown]
# ## GML

#%%
import urllib

import io
import zipfile

import matplotlib.pyplot as plt
import networkx as nx


#%%
url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"


#%%
sock = urllib.request.urlopen(url)  # open URL
s = io.BytesIO(sock.read())  # read into BytesIO "file"
sock.close()

zf = zipfile.ZipFile(s)  # zipfile object
txt = zf.read('football.txt').decode()  # read info file
gml = zf.read('football.gml').decode()  # read gml data
# throw away bogus first line with # from mejn files
gml = gml.split('\n')[1:]


#%%
G = nx.parse_gml(gml)  # parse gml data


#%%
print(txt)
# print degree for each team - number of games
for n, d in G.degree():
    print('%s %d' % (n, d))


#%%
options = {
    'node_color': 'grey',
    'node_size': 50,
    'line_color': 'grey',
    'linewidths': 0,
    'width': 0.1,
}


#%%
get_ipython().run_line_magic('pinfo', 'nx.draw_networkx')


#%%
nx.draw(G, pos=None, arrows=True, with_lables=True, **options)
plt.show()

#%% [markdown]
# ### Igraph

#%%
import igraph as ig

urllib.request.urloimport json
import urllib

url = "https://raw.githubusercontent.com/plotly/datasets/master/miserables.json"
data = []
with urllib.request.urlopen(url) as resp:
    data = json.loads(resp.read())

print(data.keys())N=len(data['nodes'])
NL=len(data['links'])
Edges=[(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]

G=ig.Graph(Edges, directed=False)

labels=[]
group=[]
for node in data['nodes']:
    labels.append(node['name'])
    group.append(node['group'])

layt=G.layout('kk', dim=3) Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
Yn=[layt[k][1] for k in range(N)]# y-coordinates
Zn=[layt[k][2] for k in range(N)]# z-coordinates
Xe=[]
Ye=[]
Ze=[]
for e in Edges:
    Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
    Ye+=[layt[e[0]][1],layt[e[1]][1], None]  
    Ze+=[layt[e[0]][2],layt[e[1]][2], None]  import plotly.plotly as py
import plotly.graph_objs as go

trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=1),
               hoverinfo='none'
               )

trace2=go.Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=dict(symbol='circle',
                             size=6,
                             color=group,
                             colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )

axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

layout = go.Layout(
         title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
         width=1000,
         height=1000,
         showlegend=False,
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text="Data source: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>[1] miserables.json</a>",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ],    )import numpy as np
import pandas as pd
import xarray
import datashader
import skimage
import holoviews as hv
import networkx as nx
from holoviews import optshv.Cycle('category2')kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
# opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))

colors = ['#000000']+hv.Cycle('Category3').values
edges_df = pd.read_csv('../assets/fb_edges.csv')
fb_nodes = hv.Nodes(pd.read_csv('../assets/fb_nodes.csv')).sort()
fb_graph = hv.Graph((edges_df, fb_nodes), label='Facebook Circles')

fb_graph.opts(cmap=colors, node_size=10, edge_line_width=1,
              node_line_color='gray', node_color='circle')from holoviews.operation.datashader import datashade, bundle_graph

bundled = bundle_graph(fb_graph)
bundled(
    datashade(
        bundled,
        normalization='linear',
        width=800,
        height=800,
    ) * bundled.nodes
).opts(
    opts.Nodes(
        color='circle',
        size=10,
        width=1000,
        cmap=colors,
        legend_position='right',
    )
)type(fig)
#%%



