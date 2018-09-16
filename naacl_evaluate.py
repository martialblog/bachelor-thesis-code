#!/usr/bin/env python3


import corpus
import evaluate
import features
import utils

from keras.models import load_model
from keras import backend as kerasbackend


# Global configuration
MAX_SENTENCE_LENGTH = 50
EMBEDDING_DIM = 300
KERAS_BATCH_SIZE = 32
WEIGHT_SMOOTHING = 0.0

embeddings = features.DummyEmbeddings(dimensions=EMBEDDING_DIM)

# Generate test Corpus object and get word embeddings for it
c_test = corpus.VUAMC('source/vuamc_corpus_test.csv', 'source/verb_tokens_test.csv', 'source/vuamc_corpus_test_pos.csv', mode='test')
c_test.validate_corpus()

# Sentences, Labels, POS Tags - I could just use better variable names
x_test, y_test, z_pos = features.generate_input_and_labels(c_test.sentences, Vectors=embeddings)

# TODO: Needs to come from the train I assume
class_weights =  list(utils.get_class_weights(c_test.label_list, WEIGHT_SMOOTHING).values())
print('loss_weight {}'.format(class_weights))

# Load model and Embeddings
model = load_model('naacl_metaphor.h5',
                   custom_objects={
                       'loss': utils.weighted_categorical_crossentropy(class_weights),
                       'f1': utils.f1,
                       'precision': utils.precision,
                       'recall': utils.recall
                   })

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
