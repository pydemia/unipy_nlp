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
import functools as ft
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


# def list_keeper_from_recursion(**kwargs):
#     def wrap(func):
#         # args_keys = func.__code__.co_varnames
#         for key, value in kwargs.items():
#             setattr(func, key, value)
#         return func
#     return wrap


# def pattern_list_keeper_from_recursion(recursive_func):

#     @ft.wraps(recursive_func)
#     def pattern_list_keeped(*args, **kwargs):
#         [
#             argname
#             for argname in recursive_func.__code__.co_varnames
#             if argname not in ['args', 'kwargs']
#         ]
#         args_keys = func.func_code.co_varnames
#         new_recursive_func = partial(recursive_func, pattern_list=)
#         return new_recursive_func(*args, **kwargs)

#     return pattern_list_keeped

def recursive_substitutor(str_or_series, pattern_list, flags=0):
    pattern_list = pattern_list.copy()
    if isinstance(str_or_series, str):
        if pattern_list:
        # if len(pattern_list) > 0:
            pattern, target = pattern_list.pop(0)
            str_or_series = re.sub(pattern, target, str_or_series, flags=flags)
            return recursive_substitutor(str_or_series, pattern_list)
        else:
            return str_or_series

    elif isinstance(str_or_series, pd.Series):
        if pattern_list:
            pattern, target = pattern_list.pop(0)
            str_or_series = str_or_series.str.replace(
                pattern, target, flags=flags,
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

    pattern_tuple = (
        [r'40(.*)41', r'(\1)'],
        [r'39(.*)', r'`\1'],
        [r'\t|\a', r' '],
        [r'(\r)', r' '],
        # [r'(\r\n)', r'\n'],
        [r'(\n)', r' '],
        [r"(([vox]{1}\\:\* |\.shape+[\s]+)\{.+\})", r''],
        [r'\b((http?|https|ftp|file)\://\S+)', r''],
        [r'\(\?\)', r' '],
        [r'[▷◇△▲▽▼★\<\>\+\-\=_·♬\|\(\)\[\]\{\}\*\'\"\~⓪①②③④⑤⑥⑦⑧⑨⑩]+', r' '],
        [r'\b(\[답변\]|\[열린소리\]|\[분실물\D*\])', r''],
        [r'^(.+)([\s]*[(from)(sent)]\: .+(subject)\:).+', _repl],
        [r'(subject\:|re\:)', r''],
        [r'(\s*\d{4} [\d]{1,2}:\d{2} [ap]{1}m[\s]{0,1}to:){0,1} (.+)', r'\2'],
        [r'(nbsp)', r' '],
        [r'([\s\D]+/[\s\w]+/(sk)[\;]*)', r''],
        [r'[\;]+', r' '],
        [r'([0-9]{2,3}\-[0-9]{2,4}\-[0-9]{4})', r''],
        [r'\S+@][a-zA-Z0-9\._]+', r''],
        [r'\b(\D+\([^\(\)]+\)/\S+/\S+[:;]*)\b', r''],
        [r'\b물류/서비스사업\b', r'물류서비스사업'],
        [r'/', r' '],
        [r'\s+', r' '],
        [r'[\^]{2,}', r'\^\^\n'],
        [r'(\B[음임함됨요(니다)])\b', r'\1\n'],
        [r'([\.\!\?]+[\s]*)', r'\n'],
    )
    # pattern_list = [(re.compile(ptn), target) for ptn, target in pattern_list]

    def _replace_list_str(row, pattern_tuple=pattern_tuple) -> list:

        if len(row) > 0:
            res = [
                recursive_substitutor(
                    normalize('NFKC', row),
                    list(pattern_tuple),
                    flags=re.IGNORECASE,
                )
                for s in row
            ]
        else:
            res = recursive_substitutor(
                normalize('NFKC', row),
                list(pattern_tuple),
                flags=re.IGNORECASE,
            )
            
        return res

    for col in raw_col_list:
        if isinstance(df[col][0], str):
            df[f'{col}{col_suffix}'] = recursive_substitutor(
                    (
                        df[col]
                        .apply(lambda s: normalize('NFKC', s))
                        .str.lower()
                    ),
                    list(pattern_tuple),
                    flags=re.IGNORECASE,
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
        r'^(\s*\d{4} [\d]{1,2}:\d{2} [ap]{1}m[\s]{0,1}to:){0,1} (.+)'
    )
    ptn_h_t = r'\2'
    ptn_i_s = r'([\s\D]+/[\s\w]+/sk[;]*)'
    ptn_i_t = r''

    ptn_tuple = (
        (ptn_a_s, ptn_a_t),
        (ptn_b_s, ptn_b_t),
        (ptn_c_s, ptn_c_t),
        (ptn_d_s, ptn_d_t),
        (ptn_e_s, ptn_e_t),
        (ptn_f_s, ptn_f_t),
        (ptn_g_s, ptn_g_t),
        (ptn_h_s, ptn_h_t),
        (ptn_i_s, ptn_i_t),
    )

    dataframe[colname_str] = (
        recursive_substitutor(
            dataframe[colname_str],
            pattern_list=list(ptn_tuple),
        )
        .str.lower()
    )

    return dataframe


def split_and_expand_str_rows(dataframe, colname_str, split_by='\n'):
    expanded_df = dataframe.drop([colname_str], axis=1).join(
        dataframe
        [colname_str]
        .str.split(split_by, expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .rename(colname_str)
    ).reset_index(drop=True)

    return (
        expanded_df
        .loc[~expanded_df['contents'].isin(["", " "]), :]
        .reset_index(drop=True)
    )


def collect_data(filepath, dump_json_ok=True):

    fpath = filepath
    dump_path = f'{fpath}/_tmp_dump'
    if not os.path.isdir(dump_path):
        os.makedirs(dump_path, exist_ok=False)
        print(f"'Results will be saved in {dump_path}")

    filepath_list = glob(f'{fpath}/saveasnew/*.xlsx')
    category_list = [
        re.findall(r'.*rawdata_(.+)_saveasnew.xlsx', s)[0]
        for s in filepath_list
    ]
    loaded_gen = (
        (category,
            read_xlsx_all_sheets(filepath)
            if 'usymphony' not in category
            else read_xlsx_usymp(filepath)
        )
        for category, filepath in zip(category_list, filepath_list)
    )
    refined_gen = (
        (category, refine_nested_excel_to_dict(excel_data))
        for category, excel_data in loaded_gen
    )
    flatted_gen = (
        {
            'table_nm': table_nm,
            'sheet_nm': sheet_nm,
            'contents': contents,
        }
        for table_nm, table_contents in refined_gen
        for sheet_nm, sheet_contents in table_contents.items()
        for contents in sum(list(sheet_contents.values()), [])
        # for contents in split_and_filter(sheet_contents.values())
    )
    res_df = pd.DataFrame(flatted_gen).drop_duplicates()
    except_cols = res_df.columns.drop('contents').tolist()
    res_df = (
        res_df
        .groupby(except_cols)
        ['contents']
        .apply(lambda x: ' '.join(x))
        .reset_index()
    )

    res_df = split_and_expand_str_rows(
        res_df,
        colname_str='contents',
        split_by=r'\n',
    )
    print(f"\nResult: {res_df.shape}")

    if dump_json_ok:
        res_df.to_json(
            f'{dump_path}/rawdata_cpred_flatted.json',
            orient='records',
            force_ascii=False,
            lines=True,
        )
    return res_df
