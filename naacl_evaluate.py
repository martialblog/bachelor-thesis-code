#!/usr/bin/env python3


import corpus
import evaluate
import features
import utils
# import numpy

from keras.models import load_model
from keras import backend as kerasbackend


# Global configuration
MAX_SENTENCE_LENGTH = 50
EMBEDDING_DIM = 300
KERAS_BATCH_SIZE = 32

# Load model and Embeddings
model = load_model('naacl_metaphor.h5',
                   custom_objects={ 'loss': utils.weighted_categorical_crossentropy([1, 32])})
# model = load_model('naacl_metaphor.h5')
embeddings = features.DummyEmbeddings(dimensions=EMBEDDING_DIM)

# Generate test Corpus object and get word embeddings for it
c_test = corpus.VUAMC('source/vuamc_corpus_test.csv', 'source/verb_tokens_test.csv', mode='test')
c_test.validate_corpus()
x_test, y_test = features.generate_input_and_labels(c_test.sentences, Vectors=embeddings)

# Generate list of label predictions for each sentence
float_predictions = model.predict(x_test, batch_size=KERAS_BATCH_SIZE)
binary_predictions = kerasbackend.argmax(float_predictions)
label_predictions = kerasbackend.eval(binary_predictions)


# Write prediction to CSV file
predictions_file = 'predictions.csv'
standard_file = 'source/verb_tokens_test_gold_labels.csv'

rows = evaluate.corpus_evaluation(c_test, label_predictions, MAX_SENTENCE_LENGTH)
evaluate.csv_evalutation(rows, predictions_file)
results = evaluate.precision_recall_f1(predictions_file, standard_file)

print(results)
