from keras.layers import Dense, Embedding, LSTM, Input, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import adam
import keras.backend as k
from keras.losses import mean_squared_error as mse
from keras.preprocessing.sequence import pad_sequences
import random
from copy import deepcopy
import numpy as np


train_batch_size = 3
epochs = 2
max_len = 3

def build_model():
    # 定义一个不限制输入的model
    inp = Input(shape=(None, ))
    x = Embedding(100, 5)(inp)
    x = LSTM(3,return_sequences=True)(x)
    x = GlobalMaxPool1D()(x)
    out = Dense(1)(x)
    opti = adam(lr=0.001)
    model = Model(inp, out)
    model.compile(optimizer=opti, loss=mse, metrics=[mse])
    return model


def shuffle_data(x, y):
    """
    按相同的规则随机打乱x和y
    """
    shuffle_seed = random.randint(0, 9999)
    random.seed(shuffle_seed)
    random.shuffle(x)
    random.seed(shuffle_seed)
    random.shuffle(y)


def mymetric(y_true, y_pred):
    """
    定义一个观测metric
    """
    return k.mean(k.pow(y_true - y_pred, 2))


def my_generator(x, y, batch_size, global_max_len):
    """
    batch内自适应生成器，pad长度根据batch内的长度自适应
    """
    n_batchs = (len(y) - 1) // batch_size + 1
    shuffle_data(x, y)
    while True:
        for batch in range(n_batchs):
            batch_x = deepcopy(x[batch * batch_size: (batch + 1) * batch_size])
            batch_y = deepcopy(y[batch * batch_size: (batch + 1) * batch_size])
            batch_max_len = len(max(batch_x, key=lambda x: len(x)))
            batch_max_len = min(batch_max_len, global_max_len)
            batch_x = pad_sequences(batch_x, batch_max_len)
            yield batch_x, batch_y

train_data = [
    [1, 2, 3, 4],
    [1, 2, 3],
    [1, 2],
    [1, 2, 4],
    [2]
]
train_y = [1, 2, 3, 4, 5]

val_data = [
    [1, 3, 2, 4],
    [1, 2, 1],
    [3, 2],
    [1, 2, 4],
    [3]
]
val_y = [1, 4, 5, 2, 3]

model = build_model()
train_g = my_generator(train_data, train_y, batch_size=train_batch_size, global_max_len=max_len)
val_g = my_generator(val_data, val_y, batch_size=train_batch_size, global_max_len=max_len)
model.fit_generator(generator=train_g,
                    steps_per_epoch=len(train_y) // train_batch_size,
                    epochs=epochs,
                    validation_data=val_g,
                    validation_steps=len(val_y) // train_batch_size)
