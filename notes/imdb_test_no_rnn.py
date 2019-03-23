import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.metrics import accuracy_score

mdb_dir = './data/aclImdb/aclImdb/'
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

    x_val = train_data[train_sample: train_sample+validation_samples]
    y_val = labels[train_sample: train_sample+validation_samples]

    test_sequences = tokenizer.texts_to_sequences(df_test.comment)
    x_test = pad_sequences(test_sequences, max_len)
    y_test = df_test.label.values

    return x_train, y_train, x_val, y_val, x_test, y_test, word_index
    
x_train, y_train, x_val, y_val, x_test, y_test, word_index = data_prepare(df_train, df_test)

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

model = Sequential()
model.add(Embedding(max_words, embeddings_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
"""
output:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_4 (Embedding)      (None, 100, 50)           2500000   
_________________________________________________________________
flatten_4 (Flatten)          (None, 5000)              0         
_________________________________________________________________
dense_7 (Dense)              (None, 32)                160032    
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 33        
=================================================================
Total params: 2,660,065
Trainable params: 2,660,065
Non-trainable params: 0
___________________________
"""
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val))
"""
Train on 15000 samples, validate on 10000 samples
Epoch 1/5
15000/15000 [==============================] - 1s - loss: 0.6176 - acc: 0.6552 - val_loss: 0.5744 - val_acc: 0.6956
Epoch 2/5
15000/15000 [==============================] - 1s - loss: 0.5161 - acc: 0.7475 - val_loss: 0.5748 - val_acc: 0.6954
Epoch 3/5
15000/15000 [==============================] - 1s - loss: 0.4740 - acc: 0.7744 - val_loss: 0.5964 - val_acc: 0.6884
Epoch 4/5
15000/15000 [==============================] - 1s - loss: 0.4303 - acc: 0.7985 - val_loss: 0.5974 - val_acc: 0.6913
Epoch 5/5
15000/15000 [==============================] - 1s - loss: 0.3757 - acc: 0.8311 - val_loss: 0.6470 - val_acc: 0.6874
"""
test_pre = model.predict_classes(x_test)
print(accuracy_score(y_test, test_pre))
# output: 0.80904

