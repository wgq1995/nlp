from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from sklearn.metrics import accuracy_score


max_features = 10000  #  选择最常见的10000个单词
maxlen = 20 #  最长句子长度
embedding_size = 8 #  词嵌入的维度

# 加载数据
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# 数据补全与截取
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print(x_train.shape) #  output: (25000, 20)

# model
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
"""
output:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 20, 8)             80000     
_________________________________________________________________
flatten_2 (Flatten)          (None, 160)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 161       
=================================================================
Total params: 80,161
Trainable params: 80,161
Non-trainable params: 0
_________________________________________________________________
"""
#  train model
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

# 查看词嵌入的取值
print(model.get_weights()[0][0]) # output: [-0.04778028 -0.13333663  0.00935533  0.00899234  0.09486715 -0.0449668
  0.11205751 -0.04313933]
# 预测
res = model.predict_classes(x_test)

print(accuracy_score(y_test, res)) # 0.76388


