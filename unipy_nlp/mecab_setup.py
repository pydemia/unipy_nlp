#!/bin/bash

git clone https://bitbucket.org/eunjeon/mecab-python-0.996
cd mecab-python-0.996
python setup.py build
python setup.py install
cd ..


git clone https://bitbucket.org/eunjeon/mecab-ko-dic



bash <(curl -s https://raw.githubusercontent.com/koshort/peunjeon/master/scripts/mecab.sh)
