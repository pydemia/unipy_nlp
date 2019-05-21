#!/bin/sh

python3 create_user_dic.py

MECAB_FILE="mecab-0.996-ko-0.9.2.tar.gz"
MEACB_LINK="mecab"

if [ -f "$MECAB_FILE"]  && [ -L "$MEACB_LINK" ]; then
   echo "The file '$MEACB_LINK' exists."
else
   echo "The file '$MEACB_LINK' in not found."
   wget "https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz"
   tar -zxf mecab-0.996-ko-0.9.2.tar.gz
   ln -s mecab-0.996-ko-0.9.2 mecab
fi

cd $MEACB_LINK

./configure --prefix="$HOME/.mecab"
make
make check
make install

cd ..


MECAB_DICT_FILE="mecab-ko-dic-2.1.1-20180720.tar.gz"
MEACB_DICT_LINK="mecab-ko-dic"

if [ -f "$MECAB_DICT_FILE"]  && [ -L "$MEACB_DICT_LINK" ]; then
   echo "The file '$MEACB_DICT_LINK' exists."
else
   echo "The file '$MEACB_DICT_LINK' in not found."
   wget "https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz"
   tar -zxf mecab-ko-dic-2.1.1-20180720.tar.gz
   ln -s mecab-ko-dic-2.1.1-20180720 mecab-ko-dic
fi

cd $MEACB_DICT_LINK

./configure --prefix="$HOME/.mecab/dic" \
 --with-mecab-config="$HOME/.mecab/bin/mecab-config" \
 --with-dicdir="$HOME/.mecab/dic" \
 --libexecdir="$HOME/.mecab/libexec/mecab"
make
make check
make install

# echo To enable dictionary, rewrite /home/jovyan/mecab/etc/mecabrc as \"dicdir = /home/jovyan/mecab/dic\"
# To enable dictionary, rewrite /home/jovyan/mecab/etc/mecabrc as "dicdir = /home/jovyan/mecab/dic"
#
# cd ..

./autogen.sh
./configure --prefix="$HOME/.mecab/dic"
make
make check
make install

./tools/add-userdic.sh

#
#
# vi mecab_home/etc/mecabrc
# dicdir = mecab_home/dic # Change dictionary directory

# git clone https://bitbucket.org/eunjeon/mecab-python-0.996
# ln -s mecab-python-0.996 mecab-python
# cd mecab-python
# python setup.py build
# python setup.py install
# cd ..

# wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
# tar -zxf mecab-ko-dic-2.1.1-20180720.tar.gz
# ln -s mecab-ko-dic-2.1.1-20180720 mecab-ko-dic
# cd mecab-ko-dic
# ./tools/add-userdic.sh
# make install
#
# cd ..
