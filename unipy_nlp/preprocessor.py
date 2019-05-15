# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:41:19 2019

@author: Youngju Jaden Kim
"""

# %%

import os
import re
import sys
import json
import random
import warnings
import itertools as it
import unidecode
from unicodedata import normalize
from glob import glob
from pprint import pprint
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# %%

import subprocess

font_dict = {
    path.split('/')[-1][:-4]: path
    for path in fm.get_fontconfig_fonts()
    if 'nanum' in path.lower().split('/')[-1]
}

for _font_nm, _font_file in font_dict.items():
    subprocess.call(['cp', _font_file, './unipy_nlp/_resources/fonts/'])



#%%

def read_xlsx(filename):
    print(f"Loading '{filename}'...")
    res = pd.read_excel(
        filename,
        sheet_name=None,
        header=None, names=['content'],
        dtype=str,
        encoding='utf-8',
        # encoding='ascii',
        # encoding='ISO-8859-1',
    )
    return res


def read_xlsx_usymp(filename):
    print(f"Loading '{filename}'...")
    sheet_name = 'Clientes'
    tmp = pd.read_excel(
        filename, # '../data/saveasnew/rawdata_usymphony_saveasnew.xlsx',
        sheet_name=sheet_name,
        # header=None, names=['content'],
        dtype=str,
        encoding='utf-8',
        # encoding='ascii',
        # encoding='ISO-8859-1',
        # na_values='nan',
        # keep_default_na=True,
    ).replace('nan', np.NaN)

    # tmp = full_df['Clientes']
    tmp.columns = [
        c.replace('Unnamed: ', 'un_')
        if 'Unnamed: ' in c
        else c
        for c in tmp.columns
    ]

    tmp['title_agg'] = tmp.iloc[:, :4].apply(
        lambda x: x.dropna().max(),
        axis=1,
    )
    tmp['reply_agg'] = (
        tmp.loc[:, tmp.columns[tmp.columns.str.contains('un_')][3:]]
        .apply(lambda x: x.dropna().max(), axis=1)
    )

    tmp['title_yn'] = tmp['조회수'].notnull()

    tmp['title'] = tmp.loc[tmp['title_yn'] == True, ['title_agg']]
    tmp['body'] = tmp.loc[tmp['title_yn'].shift(1) == True, ['title_agg']]
    tmp['body'] = tmp['body'].shift(-1)

    idx_srs = tmp['title'].dropna().index.to_series()

    idx_start = idx_srs + 2

    idx_end = idx_srs.shift(-1)
    idx_end.iloc[-1] = tmp.index[-1]

    idx_range_df = (
        pd.DataFrame(
            [idx_start, idx_end],
            index=['start', 'end']
        )
        .T
    ).astype(np.int)

    tmp['reply_idx'] = idx_range_df.apply(
        lambda x: list(range(x['start'], x['end'])),
        axis=1,
    )

    def collect_reply(df, reply_idx):
        if reply_idx not in ([], np.NaN):
            res = df.loc[reply_idx, 'reply_agg'].dropna().tolist()
        else:
            res = []
        return res

    tmp['reply'] = (
        tmp['reply_idx']
        .apply(lambda x: collect_reply(tmp, x))
        .apply(lambda x: '\n'.join(x))
    )
#     tmp['reply_joined'] = (
#         tmp['reply']
#         .apply(lambda x: '\n'.join(x))
#     )
    tmp = tmp[['title', 'body', 'reply']].dropna().reset_index(drop=True)
    # tmp['content'] = tmp.values.sum(axis=1)
    tmp['content'] = (
        pd.Series(
            tmp[tmp.columns.drop('reply')]
            .fillna('')
            .values
            .tolist()
        )
        .str.join('\n')
    )

    return OrderedDict({sheet_name: tmp})


def repl(mat, ri=re.compile(r'([\.\,\'\"]+\s*)')):
    return ri.sub(r'', ''.join(mat.group(0, 2)))
    # return ri.sub(r'', mat.group(0, 2))
    # return ri.sub(r'', mat.group(0))

def replace_str(s):
    s = normalize('NFKD', s)
    s = re.sub(r'[(re\:)(FW\:)]+', r'', s,
                flags=re.IGNORECASE,
                )
    s = re.sub(r'40(.*)41', r'(\1)', s)
    s = re.sub(r'39(.*)', r'`\1', s)
    s = re.sub(r'\t|\a', r' ', s)
    s = re.sub(r'(\r\n)|(\r)', r'\n', s)
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'\(\?\)', r' ', s)
    s = re.sub(r'[▷◇△▲▽▼★]', r' ', s)
    s = re.sub(r'[<]*[-=]{2,}[>]*', r' ', s)
    # s = re.sub(r'(\[답변\])|(\[열린소리\])|(\[분실물\D*\])', r'', s)
    s = re.sub(r'\[[(답변)|(열린소리)|(분실물\D*)]+\]', r'', s)
    # s = re.sub(r'\[(답변\])|(\[열린소리\])|(\[분실물\D*\])', r'', s)
    s = re.sub(r'물류/서비스사업', r'물류서비스사업', s)
    s = re.sub(
        r'^(.+)([\s]*[(From)(Sent)]\: .+Subject\:)(.+)', repl, s,
        flags=re.IGNORECASE,
    )
    s = normalize('NFKD', s)
    return s

def conditional_replace(row):
    if len(row) > 0:
        res = [replace_str(s) for s in row]
    else:
        res = row
    return res


def refine_content(df):

    col_list = df.columns[~df.columns.str.contains('_refined')]
    print(f'{col_list.tolist()}')
    for col in col_list:
        if isinstance(df[col][0], str):
            df[f'{col}_refined'] = (
                df[col]
                .apply(lambda s: normalize('NFKD', s))
                .str.replace(r'[(re\:)(FW\:)]+', r'', flags=re.IGNORECASE)
                .str.replace(r'40(.*)41', r'(\1)')
                .str.replace(r'39(.*)', r'`\1')
                .str.replace(r'(\t)|(\a)', r' ')
                .str.replace(r'(\r\n)|(\r)', r'\n')
                .str.replace(r'http\S+', '')
                .str.replace(r'\(\?\)', r' ')
                .str.replace(r'[▷◇△▲▽▼]', r' ')
                .str.replace(r'[<]*[-=]{2,}[>]*', r' ')
                .str.replace(r'(\[답변\])|(\[열린소리\])|(\[분실물\D*\])', r'') 
                .str.replace(r'물류/서비스사업', r'물류서비스사업')
                .str.replace(r'^(.+)([\s]*[(From)(Sent)]\: .+Subject\:)(.+)', repl, flags=re.IGNORECASE)
                .apply(lambda s: normalize('NFKD', s))
            )
        elif isinstance(df[col][0], list):
            df[f'{col}_refined'] = df[col].apply(conditional_replace)

    return df  # df[df.columns[df.columns.str.contains('_refined')]]


def loop_sheets(key, xlsx_loaded):

    if isinstance(xlsx_loaded, OrderedDict):
        for sheet in xlsx_loaded:
            print(f'{key}: {sheet}')
            xlsx_loaded[sheet] = refine_content(xlsx_loaded[sheet])

    else:
        print(f'{key}')
        xlsx_loaded = refine_content(xlsx_loaded)

    return xlsx_loaded


def rm_suffix(s):
    return s.replace('_refined', '')

