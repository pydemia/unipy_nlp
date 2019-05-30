"""Backend Support Module.
"""

# %%
import os
import matplotlib.font_manager as fm
import subprocess

font_dict = {
    path.split('/')[-1][:-4]: path
    for path in fm.get_fontconfig_fonts()
    if 'nanum' in path.lower().split('/')[-1]
}

for _font_nm, _font_file in font_dict.items():
    subprocess.call(['cp', _font_file, './unipy_nlp/_resources/fonts/'])


def build_mecab():
    cwd = os.path.abspath(os.path.dirname(__file__))
    # p = subprocess.call(
    #     [
    #         sys.executable,
    #         os.path.jon(cwd, )
    #     ]
    # )
    mecab_dir = os.path.join(
        resource_dir,
        'mecab',
    )
    output = subprocess.call(['sh', os.path.join(mecab_dir, 'build_mecab.sh')])
    print(output)


