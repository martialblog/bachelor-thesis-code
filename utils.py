#!/usr/bin/env python3


"""
General utils
"""


from os.path import exists
from shutil import copyfile
from subprocess import run
from urllib.request import urlretrieve
from zipfile import ZipFile


def download_vuamc_xml(url='http://ota.ahds.ac.uk/text/2541.zip'):
    """
    Downloads the original VUAMC.zip if necessary.
    http://ota.ahds.ac.uk/headers/2541.xml
    """

    zipped_vuamc_file = 'starterkits/2541.zip'
    unzipped_vuamc_file = 'starterkits/2541/VUAMC.xml'

    if exists(unzipped_vuamc_file):
        return

    if not exists(zipped_vuamc_file):
        try:
            print('Downloading {url}'.format(url=url))
            urlretrieve(url, zipped_vuamc_file)
        except urllib.error.HTTPError:
            print('Could not download VUAMC.zip')
            return

    zipped_vuamc = ZipFile(zipped_vuamc_file, 'r')
    zipped_vuamc.extractall('starterkits/')
    zipped_vuamc.close()
    print('Successfully extracted {url}'.format(url=url))


def generate_vuamc_csv():
    """
    Generates the CSV files used in the Shared Task, using the scripts provided by NAACL
    https://github.com/EducationalTestingService/metaphor/tree/master/NAACL-FLP-shared-task
    """

    if not exists('source/vuamc_corpus_test.csv'):
        run(['python3', 'vua_xml_parser_test.py'], cwd='starterkits')
        copyfile('starterkits/vuamc_corpus_test.csv', 'source/vuamc_corpus_test.csv')
        print('Successfully generated vuamc_corpus_test.csv')

    if not exists('source/vuamc_corpus_train.csv'):
        run(['python3', 'vua_xml_parser.py'], cwd='starterkits')
        copyfile('starterkits/vuamc_corpus_train.csv', 'source/vuamc_corpus_train.csv')
        print('Successfully generated vuamc_corpus_train.csv')
