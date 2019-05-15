"""Test Code Here.
"""


# %%

import os
import re
from glob import glob
import importlib
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

flist = glob(f'{fpath}/saveasnew/*.xlsx')
fname_list = [
    re.findall(r'.*rawdata_(.+)_saveasnew.xlsx', s)[0]
    for s in flist
]
fdict = {n: f for n, f in zip(fname_list, flist)}
ddict = {
    n: unlp.read_xlsx(f)
    if 'usymphony' not in f
    else unlp.read_xlsx_usymp(f)
    for n, f in fdict.items()
}
rdict = {k: unlp.loop_sheets(k, d) for k, d in ddict.items()}
ldict = {
    key: {
        '18년' if sheet == 'Sheet1' else sheet: (
            df.loc[:, df.columns.str.contains('_refined')]
            .rename(unlp.rm_suffix, axis=1)
            .to_dict(orient='list')
        )
        for sheet, df in xlf.items()
    }
    for key, xlf in rdict.items()
}


content_only = {
    key: sum(
        [sum(list(dic.values()), []) for sheet, dic in xlf.items()],
        [],
    )
    for key, xlf in ldict.items()
}

full_content = sum(list(content_only.values()), [])
fcont = sum([s.split('\n') for s in full_content], [])
fcont = list(
    set(filter(lambda x: x not in (r'', r'\n'), fcont))
)


#%%
ldict['usymphony']['Clientes'].keys()
# %%
ldict['ci_voice']['18년'].keys()
# %%
