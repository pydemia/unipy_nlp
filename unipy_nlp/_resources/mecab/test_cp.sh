PKG_DIR=$(pwd -P)
BASEDIR=$(dirname "$0")


mkdir -p $BASEDIR/cptest

echo $BASEDIR
echo $BASEDIR/cptest

cp $BASEDIR/add-userdic.sh $BASEDIR/cptest

