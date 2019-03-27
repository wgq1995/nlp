import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

maxlen = 50
max_features = 100000
embed_size = 300
filter_sizes = [1,2,3,5]
num_filters = 36

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Reshape((maxlen, embed_size, 1))(x)

maxpool_pool = []
for i in range(len(filter_sizes)):
    conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                 kernel_initializer='he_normal', activation='elu')(x)
    maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

z = Concatenate(axis=1)(maxpool_pool)   
z = Flatten()(z)
z = Dropout(0.1)(z)

outp = Dense(1, activation="sigmoid")(z)

model = Model(inputs=inp, outputs=outp)
model.compile(loss='binary_crossentropy', optimizer='adam')

"""
output:
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 50)           0                                            
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 50, 300)      30000000    input_2[0][0]                    
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 50, 300, 1)   0           embedding_2[0][0]                
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 50, 1, 36)    10836       reshape_2[0][0]                  
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 49, 1, 36)    21636       reshape_2[0][0]                  
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 48, 1, 36)    32436       reshape_2[0][0]                  
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 46, 1, 36)    54036       reshape_2[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 1, 1, 36)     0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 1, 1, 36)     0           conv2d_6[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 1, 1, 36)     0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_8 (MaxPooling2D)  (None, 1, 1, 36)     0           conv2d_8[0][0]                   
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 4, 1, 36)     0           max_pooling2d_5[0][0]            
                                                                 max_pooling2d_6[0][0]            
                                                                 max_pooling2d_7[0][0]            
                                                                 max_pooling2d_8[0][0]            
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 144)          0           concatenate_2[0][0]              
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 144)          0           flatten_2[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            145         dropout_2[0][0]                  
==================================================================================================
Total params: 30,119,089
Trainable params: 30,119,089
Non-trainable params: 0
__________________________________________________________________________________________________
"""
