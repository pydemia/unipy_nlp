#!/bin/sh

PKG_NAME="unipy_nlp"
RESOURCE_DIR="_resources"
MECAB_DIR="mecab"

PKG_DIR=$(pwd -P)
BASEDIR=$(dirname "$0")

cd "$BASEDIR"
echo "Working in '$BASEDIR'"

MECAB_FILE="mecab-0.996-ko-0.9.2.tar.gz"
MEACB_SRC_LINK="mecab_src"
MEACB_SRC_DIR="mecab_src"
MEACB_DIR="mecab"

# if [ -f "$MECAB_FILE"] && [ -L "$MEACB_LINK" ] && [ -e "$MEACB_LINK" ]; then
if [ -f "$MECAB_FILE" ] || [ -d "$MEACB_SRC_DIR" ]; then
  echo "'$BASEDIR/$MEACB_SRC_DIR' exists."
else
  echo "'$BASEDIR/$MEACB_SRC_DIR' is not found."
  wget "https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz"
  tar -zxf mecab-0.996-ko-0.9.2.tar.gz
  # ln -s mecab-0.996-ko-0.9.2 mecab
  mv mecab-0.996-ko-0.9.2 $MEACB_SRC_DIR
fi

cd $MEACB_SRC_DIR

export PATH="$(pwd -P):$PATH"
# ./configure --prefix="$HOME/.mecab"
./configure --prefix="$BASEDIR/$MEACB_DIR"
make
make check
make install

cd ..


MECAB_DICT_FILE="mecab-ko-dic-2.1.1-20180720.tar.gz"
MEACB_DICT_LINK="mecab-ko-dic"
MEACB_DICT_DIR="mecab-ko-dic"

# if [ -f "$MECAB_DICT_FILE"]  && [ -L "$MEACB_DICT_LINK" ] && [ -e "$MEACB_DICT_LINK" ]; then
if [ -f "$MECAB_DICT_FILE" ] || [ -L "$MEACB_DICT_DIR" ]; then
  echo "The file '$BASEDIR/$MEACB_DICT_DIR' exists."
else
  echo "The file '$BASEDIR/$MEACB_DICT_DIR' iS not found."
  wget "https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz"
  tar -zxf mecab-ko-dic-2.1.1-20180720.tar.gz
  # ln -s mecab-ko-dic-2.1.1-20180720 mecab-ko-dic
  mv mecab-ko-dic-2.1.1-20180720 $MEACB_DICT_DIR
fi

cd $MEACB_DICT_DIR
export PATH="$(pwd -P):$PATH"

# ./configure --prefix="$HOME/.mecab/dic" \
#  --with-mecab-config="$HOME/.mecab/bin/mecab-config" \
#  --with-dicdir="$HOME/.mecab/dic" \
#  --libexecdir="$HOME/.mecab/libexec/mecab"
# make
# make check
# make install

# echo To enable dictionary, rewrite /home/jovyan/mecab/etc/mecabrc as \"dicdir = /home/jovyan/mecab/dic\"
# To enable dictionary, rewrite /home/jovyan/mecab/etc/mecabrc as "dicdir = /home/jovyan/mecab/dic"
#
# cd ..

./autogen.sh
# ./configure --prefix="$HOME/.mecab/dic"
./configure --prefix="$BASEDIR/$MEACB_DIR/dic"
make
make check
make install

cd "$BASEDIR"
# USERDIC_CREATOR="./create_user_dic.py"
if [ -e "./create_user_dic.py" ]; then
  echo "\n'$BASEDIR/create_user_dic.py' exists."
  python3 create_user_dic.py
else
  echo "'$BASEDIR/create_user_dic.py' is MISSING!"
fi

cd $MEACB_DICT_DIR
bash ./tools/add-userdic.sh

cd "$PKG_DIR"

echo "Build 'MeCab' has been finished successfully."
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
