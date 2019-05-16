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

def read_xlsx_all_sheets(filename):
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


def _repl(mat, ri=re.compile(r'([\.\,\'\"]+\s*)')):
    return ri.sub(r'', ''.join(mat.group(0, 2)))
    # return ri.sub(r'', mat.group(0, 2))
    # return ri.sub(r'', mat.group(0))


def recursive_substitutor(str_or_series, pattern_list, flags=0):

    if isinstance(str_or_series, str):
        if len(pattern_list) > 0:
            pattern, target = pattern_list.pop(0)
            str_or_series = re.sub(pattern, target, str_or_series, flags=flags)
            return recursive_substitutor(str_or_series, pattern_list)
        else:
            return str_or_series

    elif isinstance(str_or_series, pd.Series):
        if len(pattern_list) > 0:
            pattern, target = pattern_list.pop(0)
            str_or_series = str_or_series.str.replace(
                pattern, target,
            )
            return recursive_substitutor(str_or_series, pattern_list, flags=flags)
        else:
            return str_or_series

    else:
        raise TypeError('`str_or_series` must be a `str` or `pandas.Series`')


def refine_content(df) -> pd.DataFrame:

    col_suffix = '_refined'
    raw_col_list = df.columns[~df.columns.str.contains(col_suffix)]
    new_col_list = raw_col_list + col_suffix
    print(f'colnames: {raw_col_list.tolist()}')

    pattern_list = [
        [r'40(.*)41', r'(\1)'],
        [r'39(.*)', r'`\1'],
        [r'\t|\a', r' '],
        [r'(\r\n)|(\r)', r'\n'],
        [r'(http\S+)', ''],
        [r'\(\?\)', r' '],
        [r'[▷◇△▲▽▼★]', r' '],
        [r'[\<]*[\-\=]{2,}[\>]*', r' '],
        [r'\[[(답변)|(열린소리)|(분실물\D*)]+\]', r''],
        [r'^(.+)([\s]*[(From)(Sent)]\: .+(Subject)\:).+', _repl],
        [r'[\_\·\♬\<\>\(\)\[\]\{\}\*\'\"\-\+\~\|⓪①②③④⑤⑥⑦⑧⑨⑩]+', r' '],
        [r'[\^]{2,}', r'\^\^\n'],
        [r'([^\s]+[음임함됨요(니다)])[\W]{1,}', r'\1\n'],
        [r'[\.{2,}\!\?]+[\s]+|[\s]{3,}', r'\n'],
        [r'[\s]{2,}', r' '],
        [r'(nbsp)', r' '],
        [r'([\s\D]+/[\s\w]+/(sk)[\;]*)', r''],
        [r'[\;]+', r' '],
        [r'[(subject\:)(re\:)]+', r''],
        [r'물류/서비스사업', r'물류서비스사업'],
        [r'\/', r' '],
        # [r'물류/서비스사업', r'물류서비스사업'],
        # [r'((\d{4} [\d]{1,2}:\d{2} [ap]{1}m[\s]{0,1}to:){0,1} .+)', r''],
        # [r'([\s\D]+/[\s\w]+/sk[;]*)', r''],
    ]
    pattern_list = [(re.compile(ptn), target) for ptn, target in pattern_list]


    def _replace_list_str(row, pattern_list=pattern_list) -> list:
        if len(row) > 0:
            res = [
                normalize(
                    # 'NFKD',
                    'NFKC',
                    recursive_substitutor(
                        s,
                        pattern_list,
                        flags=re.IGNORECASE,
                    )
                )
                for s in row
            ]
        else:
            res = row
        return res

    for col in raw_col_list:
        if isinstance(df[col][0], str):
            df[f'{col}{col_suffix}'] = (
                recursive_substitutor(df[col], pattern_list, flags=re.IGNORECASE)
                .apply(lambda s: normalize(
                    # 'NFKD',
                    'NFKC',
                    s,
                ))
                .str.lower()
            )
        elif isinstance(df[col][0], list):
            df[f'{col}{col_suffix}'] = (
                df[col]
                .apply(_replace_list_str)
                .str.lower()
            )

    res_df = df[new_col_list]
    res_df.columns = raw_col_list

    res_dict = res_df.to_dict(orient='list')
    return res_dict


def refine_nested_excel_to_dict(xlsx_loaded) -> pd.DataFrame:

    if isinstance(xlsx_loaded, OrderedDict):
        for sheet_name in xlsx_loaded.keys():
            print(f"{sheet_name}")
            xlsx_loaded[sheet_name] = refine_content(xlsx_loaded[sheet_name])
            if sheet_name == 'Sheet1':
                new_sheet_name = '19년'
                xlsx_loaded[new_sheet_name] = xlsx_loaded.pop(sheet_name)

    else:
        raise TypeError(
            '`xlsx_loaded` must be an `OrderedDict` with its sheet name.'
        )
    #     print(f'{key}')
    #     xlsx_loaded = refine_content(xlsx_loaded)

    return xlsx_loaded


def _concat_list(nested_list):
    return sum(nested_list, [])


def split_and_filter(
        iterable,
        sep='\n',
        filter_func=lambda x: x not in (r'\n', r'')
        ) -> set:
    concatted = _concat_list(list(iterable))
    splitted = _concat_list([s.split(sep) for s in concatted])
    filtered = filter(filter_func, splitted)
    # uniqued = set(filtered)
    return filtered


def refine_content_2nd(dataframe, colname_str):

    ptn_a_s = re.compile(
        r'[\_\·\♬\<\>\(\)\[\]\{\}\*\'\"\-\+\~\|⓪①②③④⑤⑥⑦⑧⑨⑩]+'  # \à\è
    )
    ptn_a_t = r' '
    ptn_b_s = re.compile(r'[\^]{2,}')
    ptn_b_t = r'\^\^\n'
    ptn_c_s = re.compile(r'([^\s]+[음임함됨요(니다)])[\W]{1,}')
    ptn_c_t = r'\1\n'
    ptn_d_s = re.compile(r'[\.{2,}\!\?]+[\s]+|[\s]{3,}')
    ptn_d_t = r'\n'
    ptn_e_s = re.compile(r'[\s]{2,}|[(nbsp)]+')
    ptn_e_t = r' '
    ptn_f_s = re.compile(r'( [\s\D]+/[\s\w]+/sk[;]*)')
    ptn_f_t = r''
    ptn_g_s = re.compile(r'[(subject:)|(re:)]+')
    ptn_g_t = r''
    ptn_h_s = re.compile(
        r'((\d{4} [\d]{1,2}:\d{2} [ap]{1}m[\s]{0,1}to:){0,1} .+)'
    )
    ptn_h_t = r''
    ptn_i_s = r'([\s\D]+/[\s\w]+/sk[;]*)'
    ptn_i_t = r''

    ptn_list = [
        (ptn_a_s, ptn_a_t),
        (ptn_b_s, ptn_b_t),
        (ptn_c_s, ptn_c_t),
        (ptn_d_s, ptn_d_t),
        (ptn_e_s, ptn_e_t),
        (ptn_f_s, ptn_f_t),
        (ptn_g_s, ptn_g_t),
        (ptn_h_s, ptn_h_t),
        (ptn_i_s, ptn_i_t),
    ]

    dataframe[colname_str] = (
        recursive_substitutor(
            dataframe[colname_str],
            pattern_list=ptn_list,
        )
        .str.lower()
    )

    return dataframe


def split_and_expand_str_rows(dataframe, colname_str, split_by='\n'):
    return dataframe.drop([colname_str], axis=1).join(
        dataframe
        [colname_str]
        .str.split(split_by, expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .rename(colname_str)
    ).reset_index(drop=True)

