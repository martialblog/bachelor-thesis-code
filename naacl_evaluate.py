#!/usr/bin/env python3


import corpus
import features
import numpy
import utils

from keras.models import load_model


# Global configuration
MAX_SENTENCE_LENGTH = 20
EMBEDDING_DIM = 50

embeddings = features.DummyEmbeddings(dimensions=EMBEDDING_DIM)

model = load_model('naacl_metaphor.model')

c_test = corpus.VUAMC('source/vuamc_corpus_test.csv', 'source/verb_tokens_test.csv', mode='test')
c_test.validate_corpus()
x, y = features.generate_input_and_labels(c_test.sentences, Vectors=embeddings)


loss_weight = 32
KERAS_LOSS = utils.weighted_categorical_crossentropy([1, loss_weight])
KERAS_OPTIMIZER = 'rmsprop'
KERAS_METRICS = ['categorical_accuracy']
KERAS_EPOCHS = 1
KERAS_BATCH_SIZE = 32


predictions = model.predict(x_test, batch_size=KERAS_BATCH_SIZE)
max_vals = kerasbackend.argmax(predictions)
label_predictions = kerasbackend.eval(max_vals)

pred_id = 0
for txt_id in c_test.tokens:
    for sentence_id in c_test.tokens[txt_id]:
        sentence = c_test.sentence(txt_id, sentence_id)
        tokens = c_test.tokens[txt_id][sentence_id]
        for tok_id, _ in enumerate(sentence):
            y_pred = label_predictions[pred_id]
            if tok_id+1 in tokens:
                print("{}_{}_{},{},{}".format(txt_id, sentence_id, tok_id+1, y_pred[tok_id % MAX_SENTENCE_LENGTH], sentence[tok_id][0]))
            if (tok_id+1) % MAX_SENTENCE_LENGTH == 0 and tok_id+1 < len(sentence):
                pred_id += 1
        pred_id += 1
