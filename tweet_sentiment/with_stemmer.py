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
from keras import optimizers
from sklearn import preprocessing
import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize
import pandas as pd
import numpy as np
import re
import sys
import seaborn as sns
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import nltk
from nltk.stem.lancaster import *
import itertools

def preprocess_text(sen):
    # lowercase all
    sen = sen.lower()

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing twitter names @nick
    sentence = re.sub(r'@\w+', ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Removing multiletters haaallloooo -> halo
    sentence = sentence.split()
    for idx, word in enumerate(sentence):
        sentence[idx] = ''.join(letter_group[0] for letter_group in itertools.groupby(word))
    sentence = ' '.join(sentence)

    return sentence


# https://www.kaggle.com/kazanova/sentiment140
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

dataset.head()
print(dataset["text"][:15])
ps = nltk.stem.WordNetLemmatizer()
print("====== Lemmatizer ======")
dataset["text"] = dataset["text"].apply(lambda x: ' '.join([ps.lemmatize(word) for word in x.split()]))
print(dataset["text"][:15])
dataset["Text Length"] = dataset["text"].str.len()
print(dataset["Text Length"].mean())

# sns.countplot(x='reviews_score', data=dataset)
# plt.show()

X = dataset.drop('reviews_score', axis=1)
y = dataset['reviews_score']

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

X1_train = []
X1_train_sentences = list(X_train["text"])
for sen in X1_train_sentences:
    X1_train.append(preprocess_text(sen))

X1_train_clean = list(X1_train)

X1_test = []
X1_test_clean = []
X1_test_sentences = list(X_test["text"])
for sen in X1_test_sentences:
    X1_test.append(preprocess_text(sen))

X1_test_clean = list(X1_test)


embeddings_dictionary = dict()
# http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
glove_file = open('/home/mati/nlp_mgr/glove/glove.twitter.27B/glove.twitter.27B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

glove_file.close()

unknown_words_in_test = set()
all_words_in_test = set()
maxlen = 50
chosen_tokenizer = 'keras'
if chosen_tokenizer == 'keras':
    """ Keras Tokenizer """
    # """
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X1_train)
    # print(tokenizer.word_index)
    X1_train = tokenizer.texts_to_sequences(X1_train)
    X1_test = tokenizer.texts_to_sequences(X1_test)
    vocab_size = len(tokenizer.word_index) + 1

    X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
    X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)

    """ Keras tokenizer embedding """
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
        else:
            unknown_words_in_test.add(word)
    # print("Ilość unik. nauczonych słów z train:", len(all_words_in_test))
    print("Nieznane słowa z test:", len(unknown_words_in_test))
    ###########
    # """
elif chosen_tokenizer == 'tbwt':
    """ TreebankWordTokenizer NLTK """
    word_index = 1
    nltk_tokenizer_word_dict = {}
    X3_train = []
    X3_test = []
    for sen in X1_train_clean:
        sentence_seq = []
        sen_tokens = word_tokenize(sen)
        for token in sen_tokens:
            if token not in nltk_tokenizer_word_dict:
                nltk_tokenizer_word_dict[token] = word_index
                word_index += 1

            sentence_seq.append(nltk_tokenizer_word_dict[token])
        X3_train.append(sentence_seq)

    vocab_size_3 = len(nltk_tokenizer_word_dict) + 1

    unknown_words_in_test = set()
    all_words_in_test = set()
    for sen in X1_test_clean:
        sentence_seq = []
        sen_tokens = word_tokenize(sen)
        for token in sen_tokens:
            all_words_in_test.add(token)
            try:
                sentence_seq.append(nltk_tokenizer_word_dict[token])
            except:
                unknown_words_in_test.add(token)
                # jesli nie znam słowa nic nie dodaje
        X3_test.append(sentence_seq)

    print("Ilość unik. nauczonych słów z train:", len(all_words_in_test))
    print("Nieznane słowa z test:", len(unknown_words_in_test))

    X3_train = pad_sequences(X3_train, padding='post', maxlen=maxlen)
    X3_test = pad_sequences(X3_test, padding='post', maxlen=maxlen)


    """ TreeBankTokenizer embedding """
    embedding_matrix_3 = zeros((vocab_size_3, 100))
    for word, index in nltk_tokenizer_word_dict.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix_3[index] = embedding_vector


    # sys.exit(0)
    ######################
elif chosen_tokenizer == 'tt':

    """ TweetTokenizer NLTK """
    word_index = 1
    tweet_tokenizer_word_dict = {}
    X2_train = []
    X2_test = []
    tweettokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    for sen in X1_train_clean:
        sentence_seq = []
        sen_tokens = tweettokenizer.tokenize(sen)
        for token in sen_tokens:
            if token not in tweet_tokenizer_word_dict:
                tweet_tokenizer_word_dict[token] = word_index
                word_index += 1

            sentence_seq.append(tweet_tokenizer_word_dict[token])
        X2_train.append(sentence_seq)

    vocab_size_2 = len(tweet_tokenizer_word_dict) + 1
    # print(tweet_tokenizer_word_dict)

    unknown_words_in_test = set()
    all_words_in_test = set()
    for sen in X1_test_clean:
        sentence_seq = []
        sen_tokens = tweettokenizer.tokenize(sen)
        for token in sen_tokens:
            all_words_in_test.add(token)
            try:
                sentence_seq.append(tweet_tokenizer_word_dict[token])
            except:
                unknown_words_in_test.add(token)
                # jesli nie znam słowa nic nie dodaje
        X2_test.append(sentence_seq)

    print("Ilość unik. nauczonych słów z train:", len(all_words_in_test))
    print("Nieznane słowa z test:", len(unknown_words_in_test))
    X2_train = pad_sequences(X2_train, padding='post', maxlen=maxlen)
    X2_test = pad_sequences(X2_test, padding='post', maxlen=maxlen)


    """ TweetTokenizer embedding """
    embedding_matrix_2 = zeros((vocab_size_2, 100))
    for word, index in tweet_tokenizer_word_dict.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix_2[index] = embedding_vector

    # print(len(embedding_matrix_2))
    # sys.exit(0)


"""
    NN Model 1 input
"""
deep_inputs = Input(shape=(maxlen,))
if chosen_tokenizer == 'keras':
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix])(deep_inputs)
elif chosen_tokenizer == 'tt':
    embedding_layer = Embedding(vocab_size_2, 100, weights=[embedding_matrix_2], trainable=False)(deep_inputs)
elif chosen_tokenizer == 'tbwt':
    embedding_layer = Embedding(vocab_size_3, 100, weights=[embedding_matrix_3], trainable=False)(deep_inputs)

LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(2, activation='softmax')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
adam = optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

print(model.summary())

plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)
if chosen_tokenizer == 'keras':
    history = model.fit(X1_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
    score = model.evaluate(X1_test, y_test, verbose=1)
elif chosen_tokenizer == 'tt':
    history = model.fit(X2_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)
    score = model.evaluate(X2_test, y_test, verbose=1)
elif chosen_tokenizer == 'tbwt':
    history = model.fit(X3_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)
    score = model.evaluate(X3_test, y_test, verbose=1)

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
