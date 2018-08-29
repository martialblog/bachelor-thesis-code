#!/usr/bin/env python3


"""
General utils
"""


from os.path import exists
from shutil import copyfile
from subprocess import run
from urllib.request import urlretrieve
from zipfile import ZipFile
from keras import backend as kerasbackend


def simplify_ratio(A, B):
    """
    Little helper to simplify ratios.
    """

    A_s = round(A / min([A, B]))
    B_s = round(B / min([A, B]))

    return_tuple = (min([A_s, B_s]), max([A_s, B_s]))

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
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= kerasbackend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = kerasbackend.clip(y_pred, kerasbackend.epsilon(), 1 - kerasbackend.epsilon())
        # calc
        loss = y_true * kerasbackend.log(y_pred) * weights
        loss = -kerasbackend.sum(loss, -1)
        return loss

    return loss


def corpus_evaluation(Corpus, predictions, max_sentence_length):

    rows = []
    pred_idx = 0

    for txt_id in Corpus.tokens:
        for sentence_id in Corpus.tokens[txt_id]:
            sentence = Corpus.sentence(txt_id, sentence_id)
            tokens = Corpus.tokens[txt_id][sentence_id]

            # Meh -.-
            if pred_idx == len(Corpus.sentences):
                break

            for tok_idx, _ in enumerate(sentence):
                labels = predictions[pred_idx]

                if tok_idx + 1 in tokens:
                    identifier = "{}_{}_{}".format(txt_id, sentence_id, tok_idx + 1)
                    word = sentence[tok_idx][0]
                    prediction = labels[tok_idx % max_sentence_length]
                    rows.append([identifier, prediction])

                if (tok_idx + 1) % max_sentence_length == 0 and tok_idx + 1 < len(sentence):
                    pred_idx += 1

            pred_idx += 1

    return rows


def csv_evalutation(rows, filename='predictions.csv'):

    with open(filename, 'w', newline='') as csvfile:
        cwriter = writer(csvfile, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
        for row in rows:
            cwriter.writerow(row)


def precision_recall(predictions_file, standard_file):

    predictions = {}
    standard = {}

    # Remove duplicate code
    with open(predictions_file, newline='') as csvfile:
        predreader = reader(csvfile, delimiter=',', quotechar='|')
        for row in predreader:
            predictions[row[0]] = int(row[1])

    with open(standard_file, newline='') as csvfile:
        stdreader = reader(csvfile, delimiter=',', quotechar='|')
        for row in stdreader:
            standard[row[0]] = int(row[1])

    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []

    for pred_idx, pred_lbl in predictions.items():
        if (pred_lbl == 1 and standard[pred_idx] == 1):
            true_pos.append(1)
        elif (pred_lbl == 0 and standard[pred_idx] == 0):
            true_neg.append(1)
        elif (pred_lbl == 0 and standard[pred_idx] == 1):
            false_pos.append(1)
        elif (pred_lbl == 1 and standard[pred_idx] == 0):
            false_neg.append(1)

    precision = sum(true_pos) / (sum(true_neg) + sum(true_pos))
    recall = sum(true_pos) / (sum(false_pos) + sum(true_pos))

    return (precision, recall)


def fscore(precision, recall):

    return 2 * (precision * recall) / (precision + recall)
