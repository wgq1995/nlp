from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mean_squared_error
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import keras.backend as K
import numpy as np


def rmse(y_true, y_pre):
    mse = mean_squared_error(y_true, y_pre)
    return K.sqrt(mse)

def build_model():
    inp = Input(shape=(1, ))
    x = Dense(16, activation="relu")(inp)
    output = Dense(1)(x)
    model = Model(inp, output)

    loss = mean_squared_error
    opti = adam(lr=0.001)
    model.compile(optimizer=opti, loss=loss, metrics=[rmse])
    return model

def get_callback(model_path):
    checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", period=1)
    earlystop = EarlyStopping(monitor="val_loss", patience=1)
    return [checkpoint, earlystop]

model = build_model()
model.summary()

x = np.random.random_sample(size=(200, 1))
y = np.array([i * 2 + 0.1 for i in x])

model_path = "./best_model.h5"
callbacs_list = get_callback(model_path)

model.fit(
    x, y,
    epochs=10, batch_size=32,
    callbacks=callbacs_list,
    validation_split=0.2
)

best_model = load_model(model_path, custom_objects={"rmse":rmse})
print(best_model.predict(np.array([0.2, 0.6])))
best_model.summary()
