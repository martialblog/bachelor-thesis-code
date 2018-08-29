#!/usr/bin/env python3


"""
Class to manage VUAMC
"""


from collections import Counter, OrderedDict
from csv import reader, DictReader
from numpy import array as nparray
from itertools import chain


class VUAMC():
    """
    Takes in the VUAMC CSV files and Token CSV training/test files
    and represents them as Objects.

    Example input files:
    $ head vuamc_corpus_train.csv
    "txt_id","sentence_id","sentence_txt"
    "a1e-fragment01","1","Latest corporate unbundler M_reveals laid-back M_approach : Roland Franklin , who is M_leading a 697m pound break-up bid for DRG , talks M_to Frank Kane"

    $ head all_pos_tokens_train_gold_labels.csv
    a1h-fragment06_116_1,0
    a1h-fragment06_116_2,0
    a1h-fragment06_116_5,1

    :param string vuamc_file: Test/Train VUAMC file as csv
    :param string tokens_file: Test/Train Tokens file as csv
    :param string mode: "train" will read labels from tokens_file
    :return: VUAMC Object
    """

    def __init__(self, vuamc_file, tokens_file, mode='train'):

        self.delimiter = ','
        self.quotechar = '"'
        self.mode = mode
        self._sentences = None
        self._token_list = None
        self._label_list = None

        self.vuamc_file = vuamc_file
        self.tokens_file = tokens_file

        self.vuamc = self._load_vuamc(self.vuamc_file)
        self.tokens = self._load_tokens(self.tokens_file)

    def _load_vuamc(self, fn):
        """
        Loads the VUAMC CSV file into an OrderedDict.

        The final structure is:
        self.vuamc['a1h-fragment06']['134']['tokens'][23]

        With the corresponding metaphor (0|1) labels:
        self.vuamc['a1h-fragment06']['134']['labels'][23]
        """

        data = OrderedDict()

        with open(fn) as csvfile:
            csvreader = DictReader(csvfile, delimiter=self.delimiter, quotechar=self.quotechar)

            for row in csvreader:
                txt_id = row['txt_id']
                sentence_id = row['sentence_id']
                sentence_txt = row['sentence_txt']

                if txt_id not in data:
                    data[txt_id] = OrderedDict()

                if txt_id in data and sentence_id in data[txt_id]:
                    exit('Identical keys in line {}'.format(csvreader.line_num))
                else:
                    data[txt_id][sentence_id] = OrderedDict()

                tokens = OrderedDict()
                labels = OrderedDict()

                for token_id, token in enumerate(sentence_txt.strip().split(' ')):
                    if token.startswith('M_'):
                        token = token[2:]
                        labels[token_id+1] = 1
                    else:
                        labels[token_id+1] = 0
                    tokens[token_id+1] = token

                data[txt_id][sentence_id]['tokens'] = tokens
                data[txt_id][sentence_id]['labels'] = labels

            return data

    def _load_tokens(self, fn):
        """
        Loads the training gold labels into an OrderedDict.
        These are used to yield the (tokens,labels) for the sentences.

        The final structure is:
        self.tokens['a1h-fragment06']['134'][23]
        """

        data = OrderedDict()

        with open(fn) as csvfile:
            csvreader = reader(csvfile, delimiter=self.delimiter, quotechar=self.quotechar)

            for row in csvreader:
                txt_id, sentence_id, token_id = row[0].split('_')

                if self.mode == 'train':
                    # TODO: I dont get this?
                    label = int(row[1])
                    label = self.vuamc[txt_id][sentence_id]['labels'][int(token_id)]
                else:
                    label = -1

                if txt_id not in data:
                    data[txt_id] = OrderedDict()

                if sentence_id not in data[txt_id]:
                    data[txt_id][sentence_id] = OrderedDict()

                if (txt_id in data and
                    sentence_id in data[txt_id] and
                    int(token_id) in data[txt_id][sentence_id]):
                    exit('Identical keys in line {}'.format(csvreader.line_num))

                data[txt_id][sentence_id][int(token_id)] = label

            return data

    def validate_corpus(self):
        """
        Check that the 'txt_id, sentence_id, token_id, class_label'-s from the csv files match.
        """

        for txt_id in self.tokens:
            for sentence_id in self.tokens[txt_id]:
                for token_id in self.tokens[txt_id][sentence_id]:
                    if self.mode == 'train':
                        assert(self.tokens[txt_id][sentence_id][token_id] ==
                               self.vuamc[txt_id][sentence_id]['labels'][token_id])
                    else:
                        assert(self.vuamc[txt_id][sentence_id]['labels'][token_id] == 0)

    def sentence(self, text_id, sentence_id):
        """
        Returns a sentence as a list of tuples (token, label) with the label from the self.tokens.
        """

        sentence = []

        for token_id in self.vuamc[text_id][sentence_id]['tokens'].keys():
            if token_id in self.tokens[text_id][sentence_id]:
                # Token is labeled as metaphor
                label = self.tokens[text_id][sentence_id][token_id]
            else:
                # Token not a metaphor
                label = 0

            sentence.append((self.vuamc[text_id][sentence_id]['tokens'][token_id], label))

        return sentence

    @property
    def sentences(self):
        """
        Yields a list of all sentences, each sentence a list of tuples (word, label).

        Example:
        [('Such', 0), ('language', 0), ('focused', 1), ('attention', 0), ('on', 0), ('the', 0), ('individuals', 0)]
        """

        def populate_sentences():
            """
            Helper to populate sentences.
            """

            for text_id in self.tokens:
                for sentence_id in self.tokens[text_id]:
                    yield self.sentence(text_id, sentence_id)

        if self._sentences is None:
            self._sentences = list(populate_sentences())

        return self._sentences

    @property
    def token_list(self):
        """
        Yields a list of all tokens
        """

        def populate_tokens():

            for sentence in self.sentences:
                yield [item[0] for item in sentence]

        if self._token_list is None:
            # Flatten list of lists
            self._token_list = list(chain(*list(populate_tokens())))

        return self._token_list

    @property
    def label_list(self):
        """
        Yields a list of all labels
        """

        def populate_labels():

            for sentence in self.sentences:
                yield [item[1] for item in sentence]

        if self._label_list is None:
            # Flatten list of lists
            self._label_list = list(chain(*list(populate_labels())))

        return self._label_list
