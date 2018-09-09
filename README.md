# A Neural Network Based Approach to Token-Level Metaphor Detection

This repository contains the source code for my bachelor thesis. It is based on the [NAACL 2018 Shared Task for Metaphor Detection](https://sites.google.com/site/figlangworkshop/shared-task) but did not compete in the task.

Contents:

 - Standalone Keras implementations for training and evaluation
 - A Jupyter Notebook with a Keras implementation and detailed descriptions
 - Utils and Helper modules for data preprocessing
 - Conversion and download tools for the VUAMC

For futher details on the Shared Task dataset, visit: https://github.com/EducationalTestingService/metaphor/tree/master/NAACL-FLP-shared-task

## Prerequisites

The following prerequisites could not be included in the repository and need to be downloaded.

### Word Embeddings

Download WordEmbeddings for encoding lexical items (Gensim KeyedVectors, or pymagnitude) into the *source/* directory.

```
cd source/
curl -O http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude
curl -O http://magnitude.plasticity.ai/word2vec+subword/GoogleNews-vectors-negative300.magnitude
```

- https://github.com/plasticityai/magnitude
- https://code.google.com/archive/p/word2vec/

### VUAMC

Download the VUAM Corpus as XML (can't be included due to copyrights) into the *starterkits/* directory.

```
cd starterkits/
curl -O http://ota.ahds.ac.uk/headers/2541.xml

# Or use the Python functions provided in the utils module
python3 -i utils.py
download_vuamc_xml()
```

The VUAMC needs to be converted into a CSV file and placed into the *source/* directory. This is done using the starterkits provided by the NAACL, which are included in the repository, or a Python function.

```
cd starterkits/
python3 vua_xml_parser.py
python3 vua_xml_parser_test.py

# Or use the Python functions provided in the utils module
python3 -i utils.py
generate_vuamc_csv()
```

## Setup

Install the requirements using pip:

``` bash
# Optional venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip3 install -r requirements.txt
```

## Start

The implementation can either run as a Jupyter Notebook. In that case just start the Notebook:

``` bash
# Start Notebook
jupyter notebook
```

Or, as a standalone Python script which can be executet as such:

``` bash
# Train the model
python3 naacl_evaluate.py

# Evaluation
python3 naacl_train.py
```
