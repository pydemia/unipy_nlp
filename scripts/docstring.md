# Docstring

```sh
pip install sphinx
pip install sphinx_rtd_theme
pip install numpydoc
pip install sphinxconctrib-napoleon

cd ~/git/unipy_nlp
mkdir docs
sphinx-apidoc -f -o docs/source unipy_nlp
cd docs
sphinx-quickstart
vim source/conf.py
vim source/index.rst

```

