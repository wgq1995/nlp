from keras.preprocessing.sequence import pad_sequences

# 从头开始截断
print(pad_sequences([[1, 2, 3]], maxlen=2, padding='pre'))
# output: [[2 3]]

# 从头开始补全
print(pad_sequences([[1, 2, 3]], maxlen=4, padding='pre'))
# output: [[0 1 2 3]]

# 从尾开始截断
print(pad_sequences([[1, 2, 3]], maxlen=2, padding='post'))
# output: [[2 3]]

# 从尾开始补全
print(pad_sequences([[1, 2, 3]], maxlen=4, padding='post'))
# output:[[1 2 3 0]]
