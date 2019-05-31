#!/bin/sh

PKG_NAME="unipy_nlp"
RESOURCE_DIR="_resources"
MECAB_DIR="mecab"

PKG_DIR=$(pwd -P)

# SCRIPT=$( readlink -m $( type -p $0 ))      # Full path to script
SCRIPT=$( readlink -f $0 )      # Full path to script

cd "$(dirname "$0")"
BASEDIR="$(pwd -P)"

# echo "PKG_DIR: $PKG_DIR"
# echo "BASH_SOURCE: "$( cd "$( echo "${BASH_SOURCE[0]%/*}" )" && pwd )""
# echo "dirname: "$(dirname "$0")""
# echo "readlink: "$(dirname $(readlink -f $0))""
echo "BASEDIR: $BASEDIR"

cd "$BASEDIR"
echo "Working in '$BASEDIR'"

MECAB_SRC="mecab-0.996-ko-0.9.2"
MECAB_FILE="$MECAB_SRC.tar.gz"
MECAB_SRC_DIR="mecab-src"
MECAB_SRC_LINK="$MECAB_SRC_DIR"
MECAB_DIR="mecab"
MECABRC_DICPATH="$MECAB_DIR/dic"

# if [ -f "$MECAB_FILE"] && [ -L "$MECAB_LINK" ] && [ -L "$MECAB_LINK" ] && [ -e "$MECAB_LINK" ]; then
# if [ -f "$MECAB_FILE" ] && [ -L "$MECAB_SRC_DIR" ]; then
if [ -d "$MECAB_SRC" ] && [ -L "$MECAB_SRC_LINK" ]; then
  echo "'$BASEDIR/$MECAB_SRC_DIR' exists."
else
  echo "'$BASEDIR/$MECAB_SRC_DIR' is not found."
  # curl -O "https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz"
  curl -L "https://bitbucket.org/eunjeon/mecab-ko/downloads/$MECAB_FILE" | tar -zx
  ln -s $MECAB_SRC $MECAB_SRC_LINK
  # tar -zxf mecab-0.996-ko-0.9.2.tar.gz
  # ln -s mecab-0.996-ko-0.9.2 mecab
  # mv mecab-0.996-ko-0.9.2 $MECAB_SRC_DIR
  # ln -s mecab-0.996-ko-0.9.2 $MECAB_SRC_DIR
fi


cd $MECAB_SRC_DIR

export PATH="$(pwd -P):$PATH"
export LD_LIBRARY_PATH="$MECAB_DIR/lib:$MECAB_DIR/libexec/mecab"
# ./configure --prefix="$BASEDIR/$MECAB_DIR"
./configure --prefix="$BASEDIR/$MECAB_DIR"

make clean
make
make check
make install

sed -i "s|^dicdir = .*$|dicdir = ${MECABRC_DICPATH}|g" $BASEDIR/$MECAB_DIR/etc/mecabrc
echo "'mecabrc' has been changed: '$MECABRC_DICPATH'"

cd ..


MECAB_DICT_SRC="mecab-ko-dic-2.1.1-20180720"
MECAB_DICT_FILE="$MECAB_DICT_SRC.tar.gz"
MECAB_DICT_DIR="mecab-ko-dic"
MECAB_DICT_LINK="$MECAB_DICT_DIR"

# if [ -f "$MECAB_DICT_FILE"]  && [ -L "$MECAB_DICT_LINK" ] && [ -e "$MECAB_DICT_LINK" ]; then
# if [ -f "$MECAB_DICT_FILE" ] && [ -L "$MECAB_DICT_DIR" ]; then
if [ -d "$MECAB_DICT_SRC" ] && [ -L "$MECAB_DICT_LINK" ]; then
  echo "The file '$BASEDIR/$MECAB_DICT_DIR' exists."
else
  echo "The file '$BASEDIR/$MECAB_DICT_DIR' iS not found."
  curl -L "https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/$MECAB_DICT_FILE" | tar -zx
  ln -s $MECAB_DICT_SRC $MECAB_DICT_LINK
  # tar -zxf mecab-ko-dic-2.1.1-20180720.tar.gz
  # ln -s mecab-ko-dic-2.1.1-20180720 mecab-ko-dic
  # mv mecab-ko-dic-2.1.1-20180720 $MECAB_DICT_DIR
  # ln -s mecab-ko-dic-2.1.1-20180720 $MECAB_DICT_DIR
fi

cd $MECAB_DICT_DIR
export PATH="$(pwd -P):$PATH"


# ./configure --prefix="$HOME/.mecab/dic" \
#  --with-mecab-config="$HOME/.mecab/bin/mecab-config" \
#  --with-dicdir="$HOME/.mecab/dic" \
#  --libexecdir="$HOME/.mecab/libexec/mecab"

./configure --prefix="$BASEDIR/$MECAB_DIR/dic" \
 --with-mecab-config="$BASEDIR/$MECAB_DIR/bin/mecab-config" \
 --with-dicdir="$BASEDIR/$MECAB_DIR/dic" \
 --libexecdir="$BASEDIR/$MECAB_DIR/libexec/mecab"
make clean
make
make check
make install


# ./autogen.sh
# # ./configure --prefix="$BASEDIR/$MECAB_DIR/dic"
# ./configure --prefix="$BASEDIR/$MECAB_DIR/dic"
# make clean
# make
# make check
# make install

# MECABRC_DICPATH="$MECAB_DIR/lib/mecab/dic/mecab-ko-dic"

# sed - i "/^dicdir = (.*)$/$MECAB_DIR\/lib\/mecab\/dic\/mecab-ko-dic" $MECAB_DIR/etc/mecabrc
sed -i "s|^dicdir = .*$|dicdir = ${MECABRC_DICPATH}|g" $BASEDIR/$MECAB_DIR/etc/mecabrc
echo "'mecabrc' has been changed: '$MECABRC_DICPATH'"

cd "$BASEDIR"
USERDIC_CREATOR="./create_user_dic.py"
if [ -e "./create_user_dic.py" ]; then
  echo "\n'$BASEDIR/create_user_dic.py' exists."
  python3 create_user_dic.py
else
  echo "'$BASEDIR/create_user_dic.py' is None."
fi


cp add-userdic.sh $MECAB_DICT_DIR/tools
cd $MECAB_DICT_DIR
bash ./tools/add-userdic.sh

sed -i "s|^dicdir = .*$|dicdir = ${MECABRC_DICPATH}|g" $BASEDIR/$MECAB_DIR/etc/mecabrc
echo "'mecabrc' has been changed: '$MECABRC_DICPATH'"

cd "$PKG_DIR"

echo "Build 'MeCab' has been finished."
