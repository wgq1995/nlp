"""
变长输入时，每个batch的输入长度不等，加快训练时间
"""
from keras.layers import Dense, Embedding, LSTM, Input, GlobalMaxPool1D
from keras.models import Model
import keras.backend as k
from keras.losses import mean_squared_error as mse
from keras.preprocessing.sequence import pad_sequences, TimeseriesGenerator
import random
import numpy as np


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


def get_batch_data(x, y, i, batch_size, global_max_len):
    """
    取出batch数据
    """
    batch_x = x[i:i + batch_size]
    batch_y = y[i:i + batch_size]
    batch_max_len = len(max(batch_x, key=lambda x: len(x)))
    batch_max_len = min(batch_max_len, global_max_len)
    batch_x = pad_sequences(batch_x, batch_max_len)
    return batch_x, batch_y

# 定义一个不限制输入的model
inp = Input(shape=(None, ))
x = Embedding(100, 5)(inp)
x = LSTM(3,return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
out = Dense(1)(x)

model = Model(inp, out)
model.compile(optimizer='adam', loss=mse, metrics=[mse])


train_batch_size = 3
epochs = 2
max_len = 3

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
for epoch in range(epochs):
    shuffle_data(train_data, train_y)
    # 训练
    pred_train = np.zeros(shape=(len(train_y), 1))
    for i in range(0, len(train_data), train_batch_size):
        cur_x, cur_y = get_batch_data(train_data, train_y, i, train_batch_size, max_len)
        model.train_on_batch(cur_x, cur_y)
        pred_train[i:i + train_batch_size] = model.predict(cur_x)

    # 每个epoch结束时候预测val的结果
    pred_val = np.zeros(shape=(len(val_y), 1))
    for j in range(0, len(val_data), train_batch_size):
        cur_x, _ = get_batch_data(val_data, val_y, j, train_batch_size, max_len)
        pred_val[j:j + train_batch_size] = model.predict(cur_x)
    print("epoch {}: train_metric: {}, val_metric: {}".
          format(epoch, k.eval(mymetric(train_y, pred_train)), k.eval(mymetric(val_y, pred_val))))

