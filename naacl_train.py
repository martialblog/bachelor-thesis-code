#!/usr/bin/env python3


import corpus
import features
import numpy
import utils
import collections

from keras.utils import to_categorical
from keras.layers import TimeDistributed, Bidirectional, LSTM, Input, Masking, Dense
from keras.models import Model
from keras import backend as kerasbackend
from sklearn.model_selection import KFold


# Global configuration
MAX_SENTENCE_LENGTH = 50
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
KFOLD_SPLIT = 5
KERAS_OPTIMIZER = 'rmsprop'
KERAS_METRICS = [utils.f1]
KERAS_EPOCHS = 1
KERAS_BATCH_SIZE = 32


print('Loading Word Embeddings')
embeddings = features.DummyEmbeddings(EMBEDDING_DIM)
# embeddings = features.Magnitudes()


# Generate training Corpus object and get word embeddings for it
c_train = corpus.VUAMC('source/vuamc_corpus_train.csv', 'source/verb_tokens_train_gold_labels.csv')
c_train.validate_corpus()
# TODO: Pass MAX_SENT_LEN
x, y = features.generate_input_and_labels(c_train.sentences, Vectors=embeddings)

# Free up some memory
del embeddings
print('Deleted Word Embeddings')

# Input data and categorical labels
x_input = x
y_labels = to_categorical(y, 2)


# Generate loss_weight, since out dataset contains 97% non-metaphor tokens
number_of_all_labels = len(c_train.label_list)
count_of_label_classes = collections.Counter(c_train.label_list)

percentage_of_non_metaphor_tokens = round(count_of_label_classes[0] / number_of_all_labels * 100)
percentage_of_metaphor_tokens = round(count_of_label_classes[1] / number_of_all_labels * 100)
ratio = utils.simplify_ratio(percentage_of_non_metaphor_tokens, percentage_of_metaphor_tokens)
print('loss_weight {}'.format(ratio))
KERAS_LOSS = utils.weighted_categorical_crossentropy(ratio)

# Create and compile model
inputs = Input(shape=(MAX_SENTENCE_LENGTH, EMBEDDING_DIM))
model = Masking(mask_value=[-1] * EMBEDDING_DIM)(inputs)
model = Bidirectional(LSTM(100, return_sequences=True, dropout=0, recurrent_dropout=0.25))(model)
outputs = TimeDistributed(Dense(2, activation='softmax'))(model)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=KERAS_OPTIMIZER, loss=KERAS_LOSS, metrics=KERAS_METRICS)


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
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1]*100)

model.save('naacl_metaphor.h5')
print('Saved model to disk')
