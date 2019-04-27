from keras.layers import Dense, Embedding, LSTM, Input, GlobalMaxPool1D
from keras.models import Model
from keras.losses import mean_squared_error as mse
from keras.preprocessing.sequence import pad_sequences


inp = Input(shape=(None, ))
x = Embedding(100, 5)(inp)
x = LSTM(3,return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
out = Dense(1)(x)

model = Model(inp, out)
model.compile(optimizer='adam', loss=mse, metrics=[mse])


batch_size = 3
epochs = 5
data = [
    [1, 2, 3, 4],
    [1, 2, 3],
    [1, 2],
    [1, 2, 4],
    [2]
]
y = [1, 2, 3, 4, 5]
for epoch in range(epochs):
    print("epoch {}:".format(epoch))
    for i in range(0, len(data), batch_size):
        cur_data = data[i:i+batch_size]
        cur_y = y[i:i+batch_size]
        max_len = len(max(cur_data, key=lambda x: len(x)))
        cur_data = pad_sequences(cur_data, max_len)
        model.train_on_batch(cur_data, cur_y)

