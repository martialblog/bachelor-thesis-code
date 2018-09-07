#!/usr/bin/env python3


"""
Module for evaluation functions
"""


from csv import writer, reader, QUOTE_MINIMAL
from collections import namedtuple


def corpus_evaluation(corpus, predictions, max_sentence_length):
    """
    Loads the predictions from the model into a printready (csv) list.

    :param Corpus corpus: VUAMC Corpus instance
    :param list predictions: List of predictions from Model
    :param int max_sentence_length: Maximum sentence length
    :return: Returns a list of CSV rows (ready to be printed), in the format: a3m-fragment02_45_21,0
    """

    rows = []
    pred_idx = 0

    for txt_id in corpus.tokens:
        for sentence_id in corpus.tokens[txt_id]:
            sentence = corpus.sentence(txt_id, sentence_id)
            tokens = corpus.tokens[txt_id][sentence_id]

            # Meh -.-
            if pred_idx == len(corpus.sentences):
                break

            for tok_idx, _ in enumerate(sentence):
                labels = predictions[pred_idx]

                if tok_idx + 1 in tokens:
                    identifier = "{}_{}_{}".format(txt_id, sentence_id, tok_idx + 1)
                    # word = sentence[tok_idx][0]
                    prediction = labels[tok_idx % max_sentence_length]
                    rows.append([identifier, prediction])

                if (tok_idx + 1) % max_sentence_length == 0 and tok_idx + 1 < len(sentence):
                    pred_idx += 1

            pred_idx += 1

    return rows


def csv_evalutation(rows, filename='predictions.csv'):
    """
    Writes the preduction of the model into a csv file (similar to the gold_labels file
    provided by NAACL.

    :param list rows: List of printable rows, such as a3m-fragment02_45_21,0
    :param string filename: Path to file to write rows in
    :return: None
    """

    with open(filename, 'w', newline='') as csvfile:
        cwriter = writer(csvfile, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
        for row in rows:
            cwriter.writerow(row)


def f1score(precision, recall):
    """
    Calculates F1 score

    :param float precision: Precision
    :param float recall: Recall
    :return: F1 Score
    """

    try:
        res = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        res = None
    except TypeError:
        res = None

    return res


def csv_to_dict(filepath):
    """
    Returns a csv file with key/values as dictionary.
    Example:
      a3m-fragment02_45_21,0
      key:a3m-fragment02_45_21
      value: 0

    :param string filepath: Path to file to load
    :return: Dictionary from csv file
    """

    ret_dict = {}

    with open(filepath, newline='') as csvfile:
        csvreader = reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            ret_dict[row[0]] = int(row[1])

    return ret_dict


def precision_recall_f1(predictions_file, standard_file):
    """
    Calculates the Precision, Recall and F1 for two csv files.
    Returns a namedtuple with the results

    Example:
    res = precision_recall_f1('predictions.csv', 'source/verb_tokens_test_gold_labels.csv')

    :param string predictions_file: File containing the predictions from the model
    :param string standard_file: File containing the gold standard from the NAACL
    :return: Precision, Recall and F1 as a namendtuple
    """

    # pylint: disable=invalid-name
    Result = namedtuple('Result', ['precision', 'recall', 'f1'])
    predictions = csv_to_dict(predictions_file)
    standard = csv_to_dict(standard_file)

    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []

    for pred_idx, pred_lbl in predictions.items():
        if (pred_lbl == 1 and standard[pred_idx] == 1):
            true_pos.append(1)
        elif (pred_lbl == 0 and standard[pred_idx] == 0):
            true_neg.append(1)
        elif (pred_lbl == 1 and standard[pred_idx] == 0):
            false_pos.append(1)
        elif (pred_lbl == 0 and standard[pred_idx] == 1):
            false_neg.append(1)

    try:
        precision = sum(true_pos) / (sum(true_pos) + sum(false_pos))
    except ZeroDivisionError:
        precision = None

    try:
        recall = sum(true_pos) / (sum(true_pos) + sum(false_neg))
    except ZeroDivisionError:
        recall = None

    result = Result(precision, recall, f1score(precision, recall))

    return result
