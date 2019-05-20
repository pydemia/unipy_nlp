
# %%
#%% [markdown]
# $$ distinctiveness(w) = \sum P(t \vert w) log\frac{P(t \vert w)}{P(w)} $$
# $$ saliency(w) = P(w) \times distinctiveness(w) $$
#
# <div align="right">(Chuang, J., 2012. Termite: Visualization techniques for assessing textual topic models)</div>

# %%
alist = ['g++', 'openjdk-7-jdk', 'python-dev', 'python3-dev']
len(pd.DataFrame(alist))

# %% ----

subprocess
pkg_list = ['g++', 'openjdk-7-jdk', 'python-dev', 'python3-dev']
#%%
os.getcwd()
# %%
pkg_list = ['g++', 'openjdk-7-jdk', 'python-dev', 'python3-dev']
os.system(
    ';'.join([
        'cd ./unipy_nlp/_resources/pkgs',
        *[f'apt-get download {pkg}' for pkg in pkg_list],
        'cd ../../../',
    ])
)
# os.system('cd ./unipy_nlp/_resources/pkgs')


# %%
subprocess

#%%
