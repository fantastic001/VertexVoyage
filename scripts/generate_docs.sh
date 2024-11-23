#!/bin/bash 

THIS_DIR=$(dirname $0)

PYTHON_PACKAGE=${1:-"vertex_voyage"}
CONFIG_EXPLANATIONS=${2:-"doc/config_explanations.yaml"}
YAML_PARSER=$THIS_DIR/yaml_parser.py
EXCLUDE_FILES="test config.py"

config_params_table="| Parameter | Explanation | Default Value |"

get_params() {
    
    # given line of python code, function outputs all parameters of it, separated by space 
    local params="$(cat /dev/stdin  | sed \
        -e 's/^.*get_config_[a-zA-Z]+//g' \
    )"
    echo $params
}

get_config_params() {
    # Get all get_config_*(key, default) calls in a given python file and return it as markdown table
    # for every key, check if explanation is available in yaml file provided 
    # if yes, add it to the table
    local python_file=$1
    echo "Processing $python_file"
    if [ ! -f $python_file ]; then
        echo "Python file $python_file not found"
        exit 1
    fi
    if echo $EXCLUDE_FILES | grep -q $python_file; then
        return 
    fi
    local config_explanations=$2
    local config_params=$(grep -P "get_config_.*" $python_file | grep -v "def get_config" \
        | grep -v "import" \
        | get_params)
    local config_params_table="| Parameter | Explanation | Default Value |"
    echo $config_params
    # for param in $config_params; do
    #     local key=$(echo $param | cut -d, -f1)
    #     local default=$(echo $param | cut -d, -f2)
    #     local explanation=$(python $YAML_PARSER $config_explanations $key)
    #     if [ -z "$explanation" ]; then
    #         explanation=""
    #     fi
    #     config_params_table="$config_params_table\n| $key | $explanation | $default |"
    # done
    # echo -e $config_params_table
}

find $THIS_DIR/../$PYTHON_PACKAGE -name "*.py" | sort | while read python_file; do
    get_config_params $python_file $CONFIG_EXPLANATIONS
done
