#!/bin/bash

readonly PROG_NAME=$(basename $0)
if [ -n "`which greadlink`" ]; then   # for macOS
    readonly PROG_DIR=$(greadlink -m $(dirname $0)) # if not installed: `brew install coreutils`
else
    readonly PROG_DIR=$(readlink -m $(dirname $0))
fi

readonly MECAB_PATH=$PROG_DIR/../../mecab

readonly DIC_PATH=$PROG_DIR/..
readonly USERDIC_PATH=${DIC_PATH}/user-dic
# readonly MECAB_EXEC_PATH=/usr/local/libexec/mecab
readonly MECAB_EXEC_PATH=$MACAB_PATH/libexec/mecab
readonly DICT_INDEX=$MECAB_EXEC_PATH/mecab-dict-index

echo $PROG_NAME
echo $PROG_DIR
echo $MACAB_PATH
echo $DIC_PATH
echo $USERDIC_PATH
echo $MECAB_EXEC_PATH
echo $DICT_INDEX

get_userdics() {
    pushd $USERDIC_PATH &> /dev/null
    echo $(ls *.csv)
    popd &> /dev/null
}

gen_cost() {
    local input_dic=$1
    echo $input_dic

    $DICT_INDEX \
        -m ${DIC_PATH}/model.def \
        -d ${DIC_PATH} \
        -u ${DIC_PATH}/user-${input_dic} \
        -f utf-8 \
        -t utf-8 \
        -a ${USERDIC_PATH}/$input_dic
}

compile() {
    pushd $DIC_PATH &> /dev/null
    make clean;make
    popd &> /dev/null
}

main() {
    echo "generating userdic..."

    for dic in $(get_userdics); do
        gen_cost $dic
    done

    compile
}

main
