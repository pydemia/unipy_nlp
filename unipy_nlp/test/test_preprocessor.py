"""Test Code Here.
"""


# %%

import os
import re
from glob import glob
import importlib
import pandas as pd
import unipy_nlp.preprocessor as unlp

importlib.reload(unlp)
importlib.reload

# %%

data_path = './data'

fpath = data_path
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
        unlp.read_xlsx_all_sheets(filepath)
        if 'usymphony' not in category
        else unlp.read_xlsx_usymp(filepath)
    )
    for category, filepath in zip(category_list, filepath_list)
)
refined_gen = (
    (category, unlp.refine_nested_excel_to_dict(excel_data))
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
    # for contents in unlp.split_and_filter(sheet_contents.values())
)
_tmp_df = pd.DataFrame(flatted_gen).drop_duplicates()
except_cols = _tmp_df.columns.drop('contents').tolist()
_tmp_df = (
    _tmp_df
    .groupby(except_cols)
    ['contents']
    .apply(lambda x: ' '.join(x))
    .reset_index()
)

# %%
# _tmp_df = unlp.refine_content_2nd(
#     _tmp_df,
#     colname_str='contents',
# )
_tmp_df = unlp.split_and_expand_str_rows(
    _tmp_df,
    colname_str='contents',
    split_by=r'\n',
)

# %%
print(_tmp_df.shape)
_tmp_df.to_json(
    f'{dump_path}/rawdata_cpred_flatted.json',
    orient='records',
    force_ascii=False,
    lines=True,
)

# %%
except_cols = _tmp_df.columns.drop('contents').tolist()
# %%
_tmp_df.groupby(_tmp_df.columns.drop('contents'))
# %%
_tmp_df.groupby(except_cols)['contents'].apply(lambda x: '\n'.join(x))

# %%
_tmp_df.index.unique()
#%%
aa = _tmp_df['contents']


#%%
aa
