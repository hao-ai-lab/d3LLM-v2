#!/bin/bash

# Displays information on how to use script
helpFunction()
{
  echo "Usage: $0 [-d small|all]"
  echo -e "\t-d small|all - Specify whether to download entire dataset (all) or just 1000 (small)"
  exit 1 # Exit script after printing help
}

# Get values of command line flags
while getopts d: flag
do
  case "${flag}" in
    d) data=${OPTARG};;
  esac
done

if [ -z "$data" ]; then
  echo "[ERROR]: Missing -d flag"
  helpFunction
fi

# Install Python Dependencies
pip install -r requirements.txt;

conda install mkl
conda install -c conda-forge faiss-cpu

# Install Environment Dependencies via `conda`
# conda install -c pytorch faiss-cpu;
conda install -c conda-forge openjdk=11;

# Download dataset into `data` folder via `gdown` command
mkdir -p data;
cd data;
if [ "$data" == "small" ]; then
  if [ ! -f "items_shuffle_1000.json" ]; then
    gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
  else
    echo "[SKIP]: items_shuffle_1000.json already exists"
  fi
  if [ ! -f "items_ins_v2_1000.json" ]; then
    gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
  else
    echo "[SKIP]: items_ins_v2_1000.json already exists"
  fi
elif [ "$data" == "all" ]; then
  if [ ! -f "items_shuffle_1000.json" ]; then
    gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
  else
    echo "[SKIP]: items_shuffle_1000.json already exists"
  fi
  if [ ! -f "items_ins_v2_1000.json" ]; then
    gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
  else
    echo "[SKIP]: items_ins_v2_1000.json already exists"
  fi
  if [ ! -f "items_shuffle.json" ]; then
    gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
  else
    echo "[SKIP]: items_shuffle.json already exists"
  fi
  if [ ! -f "items_ins_v2.json" ]; then
    gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
  else
    echo "[SKIP]: items_ins_v2.json already exists"
  fi
else
  echo "[ERROR]: argument for `-d` flag not recognized"
  helpFunction
fi
if [ ! -f "items_human_ins.json" ]; then
  gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
else
  echo "[SKIP]: items_human_ins.json already exists"
fi
cd ..

# Download spaCy large NLP model
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Build search engine index
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
# convert items.json => required doc format
python convert_product_file_format.py 
mkdir -p indexes
./run_indexing.sh
cd ..