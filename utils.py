#!/usr/bin/env python3


"""
General utils
"""


from os.path import exists
from shutil import copyfile
from subprocess import run
from urllib.request import urlretrieve
from urllib.error import HTTPError
from zipfile import ZipFile
from keras import backend as kerasbackend
from collections import Counter


def simplify_ratio(list_a, list_b):
    """
    Little helper to simplify ratios.
    """

    list_a_s = round(list_a / min([list_a, list_b]))
    list_b_s = round(list_b / min([list_a, list_b]))

    return_tuple = (min([list_a_s, list_b_s]), max([list_a_s, list_b_s]))

    return return_tuple


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
        except HTTPError:
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


def f1(y_true, y_pred):
    """
    Keras 2.0 doesn't ship the F1 Metric anymore.
    https://github.com/keras-team/keras/issues/6507
    """

    def recall(y_true, y_pred):
        """
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """

        true_positives = kerasbackend.sum(kerasbackend.round(kerasbackend.clip(y_true * y_pred, 0, 1)))
        possible_positives = kerasbackend.sum(kerasbackend.round(kerasbackend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + kerasbackend.epsilon())

        return recall

    def precision(y_true, y_pred):
        """
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """

        true_positives = kerasbackend.sum(kerasbackend.round(kerasbackend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = kerasbackend.sum(kerasbackend.round(kerasbackend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + kerasbackend.epsilon())

        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall))


def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy.
    https://github.com/keras-team/keras/issues/6261

    Variables:
    weights: numpy array of shape (C,) where C is the number of classes

    Usage:
    # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    weights = np.array([0.5,2,10])
    loss = weighted_categorical_crossentropy(weights)
    model.compile(loss=loss, optimizer='adam')
    """

    weights = kerasbackend.variable(weights)

    def loss(y_true, y_pred):
        """
        Calculate categorical_crossentropy including the weights
        """
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= kerasbackend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = kerasbackend.clip(y_pred, kerasbackend.epsilon(), 1 - kerasbackend.epsilon())
        # calc
        loss = y_true * kerasbackend.log(y_pred) * weights
        loss = -kerasbackend.sum(loss, -1)
        return loss

    return loss
