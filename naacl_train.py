#!/usr/bin/env python3


import corpus
import features
import numpy
import utils

from keras.utils import to_categorical
from keras.layers import TimeDistributed, Bidirectional, LSTM, Input, Masking, Dense
from keras.models import Model
from keras import backend as kerasbackend


# Global configuration
# TODO: Test with longer sentences
MAX_SENTENCE_LENGTH = 50
EMBEDDING_DIM = 300

KERAS_OPTIMIZER = 'rmsprop'
KERAS_METRICS = ['categorical_accuracy']
KERAS_EPOCHS = 1
KERAS_BATCH_SIZE = 32

print('Loading Word Embeddings')
# embeddings = features.Word2Vec()
embeddings = features.DummyEmbeddings(EMBEDDING_DIM)


# Generate training Corpus object and get word embeddings for it
c_train = corpus.VUAMC('source/vuamc_corpus_train.csv', 'source/verb_tokens_train_gold_labels.csv')
c_train.validate_corpus()
x, y = features.generate_input_and_labels(c_train.sentences, Vectors=embeddings)


# Generate test Corpus object and get word embeddings for it
c_test = corpus.VUAMC('source/vuamc_corpus_test.csv', 'source/verb_tokens_test.csv', mode='test')
c_test.validate_corpus()
x_test, y_test = features.generate_input_and_labels(c_test.sentences, Vectors=embeddings)

# Free up some memory
del embeddings
print('Deleted Word Embeddings')

# Input data and categorical labels
x_input = x
y_labels = to_categorical(y, 2)


# Generate Training and Validation split
indices = numpy.arange(x_input.shape[0])
numpy.random.shuffle(indices)
data = x_input[indices]
labels = y_labels[indices]
num_validation_samples = int(0.2 * x_input.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Shape of Train Data tensor:', x_train.shape)
print('Shape of Train Labels tensor:', y_train.shape)
print('Shape of Validation Data tensor:', x_val.shape)
print('Shape of validation Labels tensor:', y_val.shape)

# TODO: use different loss function
# Generate loss_weight, since out dataset contains 97% non-metaphor tokens
loss_weight = 32
# KERAS_LOSS = utils.weighted_categorical_crossentropy([1, loss_weight])
KERAS_LOSS = 'categorical_crossentropy'
print('loss_weight 1 : {}'.format(loss_weight))


# Create and compile model
inputs = Input(shape=(MAX_SENTENCE_LENGTH, EMBEDDING_DIM))
model = Masking(mask_value=[-1] * EMBEDDING_DIM)(inputs)
model = Bidirectional(LSTM(100, return_sequences=True, dropout=0, recurrent_dropout=0.25))(model)
outputs = TimeDistributed(Dense(2, activation='softmax'))(model)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=KERAS_OPTIMIZER, loss=KERAS_LOSS, metrics=KERAS_METRICS)

# Fit the model
model.fit(x_train, y_train, batch_size=KERAS_BATCH_SIZE, epochs=KERAS_EPOCHS)
scores = model.evaluate(x_val, y_val)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('naacl_metaphor.h5')
print('Saved model to disk')
