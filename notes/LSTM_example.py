import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.metrics import accuracy_score


# prepare data just like simpleRNN
print('prepare data...')
imdb_dir = './data/aclImdb/aclImdb/'
max_len = 100
train_sample = 15000
validation_samples = 10000
max_words = 50000
embeddings_dim = 50

def get_data(root_path):
    data_count = []
    data = []
    for label_type in ['neg/', 'pos/']:
        cur_paths = os.listdir(root_path + '/' + label_type)
        for path in cur_paths:
            with open(root_path + '/' + label_type + path, 'r') as f:
                data.append(f.read())
        data_count.append(len(cur_paths))
    labels = [0] * data_count[0] + [1] * data_count[1]
    res = pd.DataFrame({
        'comment': data,
        'label': labels
    })
    return res


df_train = get_data(imdb_dir + 'train')
df_test = get_data(imdb_dir + 'train')


def data_prepare(df_train, df_test):
    max_len = 100
    train_sample = 15000
    validation_samples = 10000
    max_words = 50000
    embeddings_dim = 50

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df_train.comment)
    sequences = tokenizer.texts_to_sequences(df_train.comment)
    word_index = tokenizer.word_index

    train_data = pad_sequences(sequences, max_len)
    labels = df_train.label.values

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    labels = labels[indices]

    x_train = train_data[:train_sample]
    y_train = labels[:train_sample]

    x_val = train_data[train_sample: train_sample + validation_samples]
    y_val = labels[train_sample: train_sample + validation_samples]

    test_sequences = tokenizer.texts_to_sequences(df_test.comment)
    x_test = pad_sequences(test_sequences, max_len)
    y_test = df_test.label.values

    return x_train, y_train, x_val, y_val, x_test, y_test, word_index


x_train, y_train, x_val, y_val, x_test, y_test, word_index = data_prepare(df_train, df_test)

print('load pretrained embeddings...')
embedding_path = './data/glove.6B/glove.6B.50d.txt'


def load_embedding(path):
    embeddings_index = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = line.split()
            word = value[0]
            coefs = np.asarray(value[1:], dtype=np.float32)
            embeddings_index[word] = coefs
    return embeddings_index


embeddings_index = load_embedding(embedding_path)


def get_embedding_matrix(embeddings_index, word_index, max_words, embeddings_dim):
    embedding_matrix = np.zeros((max_words, embeddings_dim), dtype=np.float32)
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


embedding_matrix = get_embedding_matrix(embeddings_index, word_index, max_words, embeddings_dim)


# build model
model = Sequential()
model.add(Embedding(max_words, embeddings_dim, input_length=max_len))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(
    x_train, y_train,
    epochs=5, batch_size=64,
    validation_data=(x_val, y_val),
    verbose=2
)

test_pre = model.predict_classes(x_test, verbose=2)
print(accuracy_score(y_test, test_pre))
