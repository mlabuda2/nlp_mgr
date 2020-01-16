from numpy import array
from numpy import asarray
from numpy import zeros
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn import preprocessing
import nltk
import pandas as pd
import numpy as np
import re
import sys
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def preprocess_text(sen):

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


yelp_reviews = pd.read_csv("/home/mati/nlpmgr/kaggle/yelp_review.csv", nrows=300)
bins = [0, 1, 3, 5]
review_names = ['bad', 'average', 'good']
yelp_reviews['reviews_score'] = pd.cut(yelp_reviews['stars'], bins, labels=review_names)

yelp_reviews.isnull().values.any()
print(yelp_reviews.shape)

yelp_reviews.head()
print(yelp_reviews["text"][3])
# sns.countplot(x='reviews_score', data=yelp_reviews)

X = yelp_reviews.drop('reviews_score', axis=1)
y = yelp_reviews['reviews_score']

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X1_train = []
X1_train_sentences = list(X_train["text"])
# X1_train_tokenized_sents = []
X1_train_tagged_sents = []
for sen in X1_train_sentences:
    tokenized_sent = nltk.word_tokenize(sen)
    # X1_train_tokenized_sents.append(tokenized_sent)
    X1_train_tagged_sents.append(nltk.pos_tag(tokenized_sent))
    X1_train.append(preprocess_text(sen))

X1_test = []
# sentences = list(X_test["text"])
X1_test_sentences = list(X_test["text"])
X1_test_tagged_sents = []
for sen in X1_test_sentences:
    tokenized_sent = nltk.word_tokenize(sen)
    X1_test_tagged_sents.append(nltk.pos_tag(tokenized_sent))
    X1_test.append(preprocess_text(sen))

""" Tagged sents - preparing data to Input 3 """
X1_train_sentences, X1_train_sentence_tags = [], []
for tagged_sentence in X1_train_tagged_sents:
    sentence, tags = zip(*tagged_sentence)
    X1_train_sentences.append(list(sentence))
    X1_train_sentence_tags.append(list(tags))

X1_test_sentences, X1_test_sentence_tags = [], []
for tagged_sentence in X1_test_tagged_sents:
    sentence, tags = zip(*tagged_sentence)
    X1_test_sentences.append(list(sentence))
    X1_test_sentence_tags.append(list(tags))

pos_tokenizer = Tokenizer()
pos_tokenizer.fit_on_texts(X1_train_sentence_tags)
pos_vocab_size = len(pos_tokenizer.word_index) + 1
X1_train_embedded_pos = pos_tokenizer.texts_to_sequences(X1_train_sentence_tags)
X1_test_embedded_pos = pos_tokenizer.texts_to_sequences(X1_test_sentence_tags)
maxlen_pos = 200
X1_train_embedded_pos = pad_sequences(X1_train_embedded_pos, padding='post', maxlen=maxlen_pos)
X1_test_embedded_pos = pad_sequences(X1_test_embedded_pos, padding='post', maxlen=maxlen_pos)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X1_train)

X1_train = tokenizer.texts_to_sequences(X1_train)
X1_test = tokenizer.texts_to_sequences(X1_test)
vocab_size = len(tokenizer.word_index) + 1
maxlen = 200

X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
glove_file = open('/home/mati/nlpmgr/glove/glove.6B/glove.6B.200d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

glove_file.close()

embedding_matrix = zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


X2_train = X_train[['useful', 'funny', 'cool']].values
X2_test = X_test[['useful', 'funny', 'cool']].values

input_1 = Input(shape=(maxlen,))
input_2 = Input(shape=(3,))
input_3 = Input(shape=(maxlen_pos,))

"""
    NN Model 3 inputs (+pos)
"""
# embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], trainable=False)(input_1)
# embedding_layer_pos = Embedding(pos_vocab_size, 10)(input_3)
# # model.add(Flatten())
# LSTM_Layer_1 = LSTM(128)(embedding_layer)
# LSTM_Layer_2 = LSTM(128)(embedding_layer_pos)
# dense_layer_1 = Dense(10, activation='relu')(input_2)
# dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
# concat_layer = Concatenate()([LSTM_Layer_1, LSTM_Layer_2, dense_layer_2])
# # concat_layer = Concatenate()([LSTM_Layer_1, dense_layer_2])
# dense_layer_3 = Dense(10, activation='relu')(concat_layer)
# dense_layer_4 = Dense(10, activation='relu')(dense_layer_3)
# output = Dense(3, activation='softmax')(dense_layer_4)
# # model = Model(inputs=[input_1, input_2], outputs=output)

# model = Model(inputs=[input_1, input_2, input_3], outputs=output)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# history = model.fit(x=[X1_train, X2_train], y=y_train, batch_size=128, epochs=13, verbose=1, validation_split=0.2)
# score = model.evaluate(x=[X1_test, X2_test], y=y_test, verbose=1)

# history = model.fit(x=[X1_train, X2_train, X1_train_embedded_pos], y=y_train, batch_size=128, epochs=13, verbose=1, validation_split=0.2)
# score = model.evaluate(x=[X1_test, X2_test, X1_test_embedded_pos], y=y_test, verbose=1)


"""
    NN Model 1 input
"""
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 50, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(3, activation='softmax')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


print(model.summary())

plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)
history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()