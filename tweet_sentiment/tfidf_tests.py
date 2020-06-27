from numpy import asarray
import itertools

import matplotlib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from keras import optimizers
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.core import Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.stem.lancaster import *
from nltk.tokenize import TweetTokenizer, word_tokenize
from numpy import asarray
from numpy import zeros
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_20newsgroups

import numpy as np

import keras

from keras.models import Model
from keras.layers import Dense, Activation, concatenate, Embedding, Input

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# https://www.kaggle.com/kazanova/sentiment140
# https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis
dataset = pd.read_csv("/home/mati/nlp_mgr/tweet_sentiment/training.1600000.processed.noemoticon.csv",
                         header=None,
                         # skiprows=790000, nrows=20000,
                         # skiprows=780000, nrows=40000,
                         skiprows=650000, nrows=300000,
                        #  skiprows=500000, nrows=600000,
                         encoding='latin1')
dataset.columns = ['target','ids','date','flag','user','text']
bins = [-1, 2, 4]
review_names = ['bad', 'good']
dataset['reviews_score'] = pd.cut(dataset['target'], bins, labels=review_names)

dataset.isnull().values.any()
print(dataset.shape)

print("====== Lemmatizer ======")
ps = nltk.stem.WordNetLemmatizer()
dataset["text"] = dataset["text"].apply(lambda x: ' '.join([ps.lemmatize(word) for word in x.split()]))

X = dataset.drop('reviews_score', axis=1)
y = dataset['reviews_score']

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

X_train = list(X_train["text"])
X_test = list(X_test["text"])

vectorizer = TfidfVectorizer(max_features=300)
vectorizer = vectorizer.fit(X_train)

df_train = vectorizer.transform(X_train)
df_test = vectorizer.transform(X_test)
# df_test = vectorizer.transform(X_test).toarray()
# df_test = vectorizer.transform(X_test).toarray()
# df_train = df_train[:, :, None] #//shape - (3,6,1)
# df_test = df_test[:, :, None] #//shape - (3,6,1)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

maxlen = 50

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_train = pad_sequences(sequences_train, padding='post', maxlen=maxlen)

sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_test = pad_sequences(sequences_test, padding='post',  maxlen=maxlen)

vocab_size = len(tokenizer.word_index) + 1
embedding_size = 300

input_tfidf = Input(shape=(300,))
# input_tfidf = Input(shape=df_train.shape[1:])
input_text = Input(shape=(maxlen,))

embedding = Embedding(vocab_size, embedding_size, input_length=maxlen)(input_text)

# this averaging method taken from:
# https://stackoverflow.com/a/54217709/1987598

# mean_embedding = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1))(embedding)
# concatenated = concatenate([input_tfidf, mean_embedding])
dropout_layer = Dropout(0.1)(input_tfidf)
dense1 = Dense(256, activation='relu')(dropout_layer)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(2, activation='softmax')(dense2)
model = Model(inputs=input_tfidf, outputs=dense3)
# model = Model(inputs=[input_tfidf, input_text], outputs=dense3)

# model 0
# dropout_layer = Dropout(0.1)(input_tfidf)
# LSTM_Layer_1 = LSTM(128, dropout=0.15, recurrent_dropout=0.15)(input_tfidf)
# dense_layer_1 = Dense(64, activation='relu')(LSTM_Layer_1)
# dense_layer_2 = Dense(2, activation='softmax')(dense_layer_1)
# model = Model(inputs=input_tfidf, outputs=dense_layer_2)

model.summary()

adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])


# history = model.fit([df_train, sequences_train], y_train, batch_size=128, epochs=7, verbose=1, validation_split=0.2)
# score = model.evaluate([df_test, sequences_test], y_test, verbose=1)

plot_model(model, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)


history = model.fit(df_train, y_train, batch_size=128, epochs=7, verbose=1, validation_split=0.2)
score = model.evaluate(df_test, y_test, verbose=1)

print("Loss:", score[0])
print("Accuracy:", score[1])

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
