# A Neural-Network-Based Approach to Token-Level Metaphor Detection

This repository contains the source code for my bachelor thesis on metaphor detection using Neural Networks. It is based on the [NAACL 2018 Shared Task for Metaphor Detection](https://sites.google.com/site/figlangworkshop/shared-task) but did not compete in the task.

Contents:

 - A Jupyter Notebook with a Keras implementation and detailed descriptions
 - Standalone Keras implementations for training and evaluation
 - Utils and Helper modules for data preprocessing
 - Conversion and download tools for the VUAMC (training data)

For further details on the Shared Task and the training data, visit: https://github.com/EducationalTestingService/metaphor

## Prerequisites

The following prerequisites could not be included in the repository and need to be downloaded.

### Word Embeddings

Download Word Embeddings for encoding lexical items (Gensim KeyedVectors, or pymagnitude) into the *source/* directory.

```
cd source/
curl -O http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude
curl -O http://magnitude.plasticity.ai/word2vec+subword/GoogleNews-vectors-negative300.magnitude
```

- https://github.com/plasticityai/magnitude
- https://code.google.com/archive/p/word2vec/

### VUAMC

Download the VUAM Corpus as XML (can't be included due to its license) into the *starterkits/* directory.

```
cd starterkits/
curl -O http://ota.ahds.ac.uk/headers/2541.xml

# Or use the Python functions provided in the utils module
python3 -i utils.py
download_vuamc_xml()
```

The VUAMC needs to be converted into a CSV file and placed into the *source/* directory. This is done using the starterkits scripts provided by the NAACL, which are included in the repository, or a Python function.

```
cd starterkits/
python3 vua_xml_parser.py
python3 vua_xml_parser_test.py

# Or use the Python functions provided in the utils module
python3 -i utils.py
generate_vuamc_csv()
```

## Setup

Install the Python 3 requirements using pip:

``` bash
# Optional venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip3 install -r requirements.txt
```

## Start

The implementation can either run as a Jupyter Notebook - in this case just start the Notebook:

``` bash
# Start Notebook
jupyter notebook
```

or, as a standalone Python script which can be executed as such:

``` bash
# Train the model
python3 naacl_train.py

# Evaluation
python3 naacl_evaluate.py
```
