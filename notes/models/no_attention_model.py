from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layer
import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

embed_size = 300 # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(GRU(64, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
conc = Dense(64, activation="relu")(conc)
conc = Dropout(0.1)(conc)
outp = Dense(1, activation="sigmoid")(conc)

model = Model(inputs=inp, outputs=outp)
model.compile(loss='binary_crossentropy', optimizer='adam')

model.summary()

"""
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 50)            0                                            
____________________________________________________________________________________________________
embedding_1 (Embedding)          (None, 50, 300)       30000000    input_1[0][0]                    
____________________________________________________________________________________________________
bidirectional_1 (Bidirectional)  (None, 50, 128)       140160      embedding_1[0][0]                
____________________________________________________________________________________________________
global_average_pooling1d_1 (Glob (None, 128)           0           bidirectional_1[0][0]            
____________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalMa (None, 128)           0           bidirectional_1[0][0]            
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 256)           0           global_average_pooling1d_1[0][0] 
                                                                   global_max_pooling1d_1[0][0]     
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 64)            16448       concatenate_1[0][0]              
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 64)            0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1)             65          dropout_1[0][0]                  
====================================================================================================
Total params: 30,156,673
Trainable params: 30,156,673
Non-trainable params: 0
____________________________________________________________________________________________________
数据流动过程：
1. 原始的数据经过embedding层， (50, ) ==> (50, 300)
2. 经过双向的GRU层， (50, 300) ==> (50, 64 + 64), 参数个数： 2 × 3 × （300 + 64 + 1） × 64 = 140160
3. 进行全局最大和全局平均， 并且concatenate两者结果， (50, 128) ==> (128 + 128, )
4. 全连接层1， (256, ) ==> (64, )， 参数个数： (256 + 1) × 64 = 16448
5. 全连接层2, (64, ) ==> (1, ), 参数个数: (64 + 1) × 1 = 65
"""
