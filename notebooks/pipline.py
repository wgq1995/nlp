import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
import gc
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, SimpleRNN
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

embed_size = 300
max_features = 5000
maxlen = 50
n_train = 30000
n_test = 2000
train_path = './data/train.csv'
test_path = './data/test.csv'
embedding_path = './data/glove.840B.300d.txt'

contraction_mapping = {"What's":"What is", "ain't":"is not", "aren't":"are not","can't":"cannot",
                       "'cause":"because", "could've":"could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]


def clean_text(text):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    words = text.split(' ')
    res = []
    for i in range(len(words)):
        if words[i] in contraction_mapping:
            res += contraction_mapping[words[i]].split(' ')
        else:
            res.append(words[i])
    res = ' '.join(res)
    for punct in puncts:
        res = res.replace(punct, f' {punct} ')
    return res

# read
df_train = pd.read_csv(train_path, nrows=n_train)
df_test = pd.read_csv(test_path, nrows=n_test)

#clear
df_train['question_text'] = df_train['question_text'].apply(clean_text)
df_test['question_text'] = df_test['question_text'].apply(clean_text)

def prepare_data(max_features, df_train, df_test):
    # define tk,use train and test data
    tk = Tokenizer(num_words=max_features)
    tk.fit_on_texts(np.concatenate((df_train.question_text.values, df_test.question_text.values)))
    # tk data
    df_train, df_val = train_test_split(df_train, test_size=0.05, random_state=42)
    train_x = tk.texts_to_sequences(df_train.question_text.values)
    val_x = tk.texts_to_sequences(df_val.question_text.values)
    test_x = tk.texts_to_sequences(df_test.question_text.values)
    word_index = tk.word_index
    # pad data
    train_x = pad_sequences(train_x, maxlen=maxlen)
    val_x = pad_sequences(val_x, maxlen=maxlen)
    test_x = pad_sequences(test_x, maxlen=maxlen)
    # get label
    train_y = df_train.target.values
    val_y = df_val.target.values
    return train_x, val_x, test_x, train_y, val_y, word_index

train_x, val_x, test_x, train_y, val_y, word_index = prepare_data(max_features, df_train, df_test)
print(len(word_index))

del df_train, df_test
gc.collect()

embeddings_index = {}
with open(embedding_path) as f:
    for line in tqdm(f):
        values = line.split(' ') # 要用split(' '),不能用split()
        word = values[0]
        if word in word_index:
            vector = np.asarray(values[1:], dtype=np.float16)
            embeddings_index[word] = vector
print(len(embeddings_index))

embedding_matrix = np.zeros((max_features, embed_size))
for word, i in tqdm(word_index.items()):
    if i < max_features:
        embedding_vector = embeddings_index.get(word) # 这里用get方法，不存在返回None，直接用key不存在会报错
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

del embeddings_index, word_index
gc.collect()

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x = Bidirectional(LSTM(64))(x)
x = Dense(32, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(train_x, train_y, batch_size=64, epochs=5, validation_data=(val_x, val_y))

