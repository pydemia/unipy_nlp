"""Test Code Here.
"""

# %%

import os
import re
from glob import glob
import importlib
import pandas as pd
import unipy_nlp.data_collector as udcl

importlib.reload(udcl)

# %% Variables

# data_path = './data'

# fpath = data_path
# dump_path = f'{fpath}/_tmp_dump'
# if not os.path.isdir(dump_path):
#     os.makedirs(dump_path, exist_ok=False)
#     print(f"'Results will be saved in {dump_path}")

# filepath_list = glob(f'{fpath}/saveasnew/*.xlsx')
# category_list = [
#     re.findall(r'.*rawdata_(.+)_saveasnew.xlsx', s)[0]
#     for s in filepath_list
# ]


# # %% Data Loading: from Excel
# loaded_gen = (
#     (
#         category,
#         udcl.read_xlsx_all_sheets(filepath)
#     (category,
#         udc.read_xlsx_all_sheets(filepath)
#         if 'usymphony' not in category
#         else udcl.read_xlsx_usymp(filepath)
#     )
#     for category, filepath in zip(category_list, filepath_list)
# )

# # %% Data Refining 1:
# # Substitute Strings by cell, then Return it as dict
# refined_gen = (
#     (category, udcl.refine_nested_excel_to_dict(excel_data))
#     for category, excel_data in loaded_gen
# )

# # %% Data Flattening:
# # Nested File(table), Sheet, Column -> Flat Dicts
# # Compatible with JSON & pandas.DataFrame
# flatted_gen = (
#     {
#         'table_nm': table_nm,
#         'sheet_nm': sheet_nm,
#         'contents': contents,
#     }
#     for table_nm, table_contents in refined_gen
#     for sheet_nm, sheet_contents in table_contents.items()
#     for contents in sum(list(sheet_contents.values()), [])
# )

# # %% Data Refining 2:
# # Drop Duplicates & Concat Strings groupby (table, sheet)
# _tmp_df = pd.DataFrame(flatted_gen).drop_duplicates()


# # %% Concatting & Splitting `contents`:
# # For preventing sentences from `invalid carrige return`s
# except_cols = _tmp_df.columns.drop('contents').tolist()
# _tmp_df = (
#     _tmp_df
#     .groupby(except_cols)
#     ['contents']
#     .apply(lambda x: ' '.join(x))
#     .reset_index()
# )

# _tmp_df = udcl.split_and_expand_str_rows(
#     _tmp_df,
#     colname_str='contents',
#     split_by=r'\n',
# )


# # %% Dump data to JSON format
# _tmp_df.to_json(
#     f'{dump_path}/prep/rawdata_cpred_flatted.json',
#     orient='records',
#     force_ascii=False,
#     lines=True,
# )
# _tmp_df['contents'].apply(len).hist()

# %% ALL IN ONE

_tmp_df, _tmp_filename = udcl.collect_data(
    './data',
    dumppath='./data/_tmp_dump/prep',
    dump_json_ok=True,
    return_tuple=True,
)



#%%
