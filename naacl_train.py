#!/usr/bin/env python3


import corpus
import features
import utils

import collections
import numpy
from keras.utils import to_categorical, plot_model
from keras.layers import TimeDistributed, Bidirectional, LSTM, Input, Masking, Dense
from keras.models import Model
from keras import backend as kerasbackend
from sklearn.model_selection import KFold


# Global configuration
MAX_SENTENCE_LENGTH = 50
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
WEIGHT_SMOOTHING = 0.0
KFOLD_SPLIT = 2
KERAS_OPTIMIZER = 'rmsprop'
KERAS_METRICS = [utils.precision, utils.recall]
KERAS_EPOCHS = 1
KERAS_BATCH_SIZE = 32
KERAS_DROPOUT = 0.25
KERAS_ACTIVATION = 'softmax'

print('Loading Word Embeddings')
# embeddings = features.Magnitudes()
# embeddings = features.Word2Vec()
embeddings = features.DummyEmbeddings(EMBEDDING_DIM)


# Generate training Corpus object and get word embeddings for it
c_train = corpus.VUAMC('source/vuamc_corpus_train.csv', 'source/verb_tokens_train_gold_labels.csv')
c_train.validate_corpus()

x, y = features.generate_input_and_labels(c_train.sentences, Vectors=embeddings, max_len=MAX_SENTENCE_LENGTH)

# Free up some memory
del embeddings
print('Deleted Word Embeddings')

# Input data and categorical labels
x_input = x
y_labels = to_categorical(y, 2)

# Average sentence length ()
mean_sentence_len = numpy.mean([len(sent) for sent in c_train.sentences]) # 20.020576654388417
variance_sentence_len = numpy.var([len(sent) for sent in c_train.sentences]) # 192.96947371802224
stdeviation_sentence_len = numpy.std([len(sent) for sent in c_train.sentences]) # 13.891345281074193

# Generate loss_weight, since out dataset contains 97% non-metaphor tokens
number_of_all_labels = len(c_train.label_list)
count_of_label_classes = collections.Counter(c_train.label_list)

class_weights =  list(utils.get_class_weights(c_train.label_list, WEIGHT_SMOOTHING).values())
print('loss_weight {}'.format(class_weights))
KERAS_LOSS = utils.weighted_categorical_crossentropy(class_weights)

# Create and compile model
inputs = Input(shape=(MAX_SENTENCE_LENGTH, EMBEDDING_DIM), name='sentence_input')
model = Masking(mask_value=[-1] * EMBEDDING_DIM, name='masking_padding')(inputs)
model = Bidirectional(LSTM(100, return_sequences=True, dropout=0, recurrent_dropout=KERAS_DROPOUT), name='hidden_lstm')(model)
outputs = TimeDistributed(Dense(2, activation=KERAS_ACTIVATION), name='labels_output')(model)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=KERAS_OPTIMIZER, loss=KERAS_LOSS, metrics=KERAS_METRICS)

# Requires pydot and graphviz to be installed
plot_model(model, to_file='naacl_metaphor_model.png', show_shapes=True)

# Generate Training and Validation split
kfold = KFold(n_splits=KFOLD_SPLIT, shuffle=True, random_state=1337)
for train, test in kfold.split(x_input, y_labels):
    x_train = x_input[train]
    x_val = x_input[test]
    y_train = y_labels[train]
    y_val = y_labels[test]

    # Fit the model for each split
    model.fit(x_train, y_train,
              batch_size=KERAS_BATCH_SIZE,
              epochs=KERAS_EPOCHS,
              validation_data=(x_val, y_val))

    scores = model.evaluate(x_val, y_val)
    print('Loss: {:.2%}'.format(scores[0]))
    print('Precision: {:.2%}'.format(scores[1]))
    print('Recall: {:.2%}'.format(scores[2]))

model.save('naacl_metaphor.h5')
print('Saved model to disk')
