#!/usr/bin/env python3


"""
Class to manage VUAMC
"""


from collections import Counter, OrderedDict
from sys import exit
import argparse
import csv
import logging
import numpy


class Corpus():
    """
    Represent flpst files like:

    --- 8< --- vuamc_corpus_train.csv --- 8< ---
    "txt_id","sentence_id","sentence_txt"
"a1e-fragment01","1","Latest corporate unbundler M_reveals laid-back M_approach : Roland Franklin , who is M_leading a 697m pound break-up bid for DRG , talks M_to Frank Kane"
"a1e-fragment01","2","By FRANK KANE"
    ...
    --- >8 ---

    --- 8< --- all_pos_tokens.csv --- 8< ---
    a1h-fragment06_114_1,0
    a1h-fragment06_114_3,0
    a1h-fragment06_114_4,0
    a1h-fragment06_115_2,0
    a1h-fragment06_115_3,0
    a1h-fragment06_116_1,0
    a1h-fragment06_116_2,0
    a1h-fragment06_116_5,1
    --- >8 ---

    and their 'test' counterparts without class labels (or 'M_'-information).

    Parameters:
        vuamc_fn: file name string
        tokens_fn: file name string
        mode: "train" will read labels from tokens_fn
    """

    def __init__(self, vuamc_fn, tokens_fn, mode="train", sanity_check=False):

        self.log = logging.getLogger(type(self).__name__)
        self.vuamc_delimiter = self.tokens_delimiter = ","
        self.vuamc_quotechar = self.tokens_quotechar = '"'

        # set initial values for params
        self.vuamc_fn = vuamc_fn
        self.tokens_fn = tokens_fn
        self.mode = mode

        # load files
        self.vuamc = self._load_vuamc(self.vuamc_fn)
        self.tokens = self._load_tokens(self.tokens_fn)

        if sanity_check:
            self._sanity_check()

        self._sentences = None

    def _load_vuamc(self, fn):
        """
        self.vuamc['a1h-fragment06']['134']['tokens'][23]
        self.vuamc['a1h-fragment06']['134']['labels'][23]
        """

        data = OrderedDict()
        with open(fn) as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=self.vuamc_delimiter,
                                       quotechar=self.vuamc_quotechar)
            for row in csvreader:
                txt_id = row['txt_id']
                sentence_id = row['sentence_id']
                sentence_txt = row['sentence_txt']

                if txt_id not in data:
                    data[txt_id] = OrderedDict()

                if txt_id in data and sentence_id in data[txt_id]:
                    exit("Identical keys in line {}".format(csvreader.line_num))
                else:
                    data[txt_id][sentence_id] = OrderedDict()

                tokens = OrderedDict()
                labels = OrderedDict()
                for token_id, token in enumerate(
                        sentence_txt.strip().split(" ")):
                    if token.startswith("M_"):
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
        self.tokens['a1h-fragment06']['134'][23]
        """

        data = OrderedDict()
        with open(fn) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=self.tokens_delimiter,
                                   quotechar=self.tokens_quotechar)
            for row in csvreader:
                txt_id, sentence_id, token_id = row[0].split('_')

                if self.mode == "train":
                    label = int(row[1])
                    # ?FIXME:
                    # here, we are using the additional information from the
                    # filtered stop words. still, we are only using the test
                    # partition so, all in all we should be on the safe side.
                    label = self.vuamc[txt_id][sentence_id]['labels'][int(token_id)]
                else:
                    label = -1

                if txt_id not in data:
                    data[txt_id] = OrderedDict()

                if sentence_id not in data[txt_id]:
                    data[txt_id][sentence_id] = OrderedDict()

                if (txt_id in data and sentence_id in data[txt_id] and
                        int(token_id) in data[txt_id][sentence_id]):
                    exit("Identical keys in line {}".format(csvreader.line_num))

                data[txt_id][sentence_id][int(token_id)] = label

            return data

    def _sanity_check(self):
        """
        Check that the 'txt_id, sentence_id, token_id, class_label'-s from the
        files:
            - vuamc_corpus.csv
            - ..._tokens.csv
        match.
        """

        for txt_id in self.tokens:
            for sentence_id in self.tokens[txt_id]:
                for token_id in self.tokens[txt_id][sentence_id]:
                    self.log.info(
                        "%s %d %d", " ".join([str(x) for x in [txt_id,
                                                               sentence_id,
                                                               token_id]]),
                        self.tokens[txt_id][sentence_id][token_id],
                        self.vuamc[txt_id][sentence_id]['labels'][token_id])
                    assert (self.tokens[txt_id][sentence_id][token_id] ==
                            self.vuamc[txt_id][sentence_id]['labels'][token_id])

    def sentence(self, txt_id, sentence_id):
        sentence = []
        for token_id in self.vuamc[txt_id][sentence_id]['tokens'].keys():
            if token_id in self.tokens[txt_id][sentence_id]:
                label = self.tokens[txt_id][sentence_id][token_id]
            else:
                label = 0
            sentence.append((self.vuamc[txt_id][sentence_id]['tokens'][token_id],
                             label))
        return sentence

    @property
    def sentences(self):
        """ Yield list (sentences) of tuples (word, label). """

        def populate_sentences():
            """ Helper to populate sentences. """

            for txt_id in self.tokens:
                for sentence_id in self.tokens[txt_id]:
                    yield self.sentence(txt_id, sentence_id)

        if self._sentences is None:
            self._sentences = list(populate_sentences())

        return self._sentences

    @staticmethod
    def X_y_sentence(model, sentence, maxlen=None):

        retval_X = []
        retval_y = []

        # ..._pads: list of padded entried - multiple if len(toks) > max_len
        sentence_toks_pads = Utils.pad_toks([tok[0] for tok in sentence],
                                            maxlen=maxlen)
        sentence_ys_pads = Utils.pad_toks([tok[1] for tok in sentence],
                                          value=0, maxlen=maxlen)

        for tmp_id, sentence_toks_pad in enumerate(sentence_toks_pads):
            sentence_retval = []
            sentence_retval.extend(Utils.toks2feat(sentence_toks_pad, model))
            retval_X.append(numpy.array(sentence_retval))
            retval_y.append(numpy.array(sentence_ys_pads[tmp_id]))

        return retval_X, retval_y

    def X_y(self, model, sentence=None, maxlen=None):

        retval_X = []
        retval_y = []

        if sentence is None:
            sentences = self.sentences
        else:
            sentences = [sentence]

        for sent in sentences:
            X, y = Corpus.X_y_sentence(model, sent, maxlen=maxlen)
            retval_X.extend(X)
            retval_y.extend(y)

        return retval_X, retval_y

    def X(self, model):

        X, _ = self.X_y(model)

        return X

    def y(self, model):

        _, y = self.X_y(model)

        return y

    # def Xposs(self, maxlen=None):
    #     retval = []
    #     for sent in self.sentences:
    #         poss = [tag[1] for tag in pos_tag([tok[0] for tok in sent])]
    #         sentence_poss_pads = Utils.pad_toks(poss,
    #                                             value=0,
    #                                             maxlen=maxlen)
    #         for tmp_id, sentence_poss_pad in enumerate(sentence_poss_pads):
    #             retval.append(Utils.poss2feat(sentence_poss_pad))
    #     return numpy.array(retval)
