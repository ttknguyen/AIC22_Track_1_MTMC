#!/bin/bash

# Add conda-forge channel
conda config --append channels conda-forge

# Update base env
conda update --a --y
conda clean --a --y

# Absolute path to this script, e.g. /MLKit/setup/create.sh
script_path=$(readlink -f "$0")
# Absolute path this script is in, thus /MLKit/setup
dirpath=$(dirname "$script_path")
yml_path="${dirpath}/mlkit_py3.9_torch1.10.1.yml"

# Create mlkit env
conda env create -f "${yml_path}" --y
conda activate mlkit
conda update --a --y
