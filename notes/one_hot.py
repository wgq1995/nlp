import numpy as np
from keras.preprocessing.text import Tokenizer


def my_one_hot_encode(data, max_len=5):
    token_index = {}
    word_id = 1
    for sample in data:
        for word in sample.split(' '):
            if word not in token_index:
                token_index[word] = word_id
                word_id += 1

    res = np.zeros(shape=(len(data), max_len))
    for i, sample in enumerate(samples):
        for j, word in enumerate(sample.split(' ')):
            if j == max_len:
                break
            word_index = token_index.get(word)
            res[i][j] = word_index
    return token_index, res


def keras_one_hot_encode(data, max_len=5):
    tokenizer = Tokenizer(max_len)  # 前max_len个出现频率最高的单词
    tokenizer.fit_on_texts(data)

    sequences = tokenizer.texts_to_sequences(data)  # 得到句子中每个单词的编码位置

    word_index = tokenizer.word_index
    return word_index, sequences


if __name__ == '__main__':
    samples = [
        "I like I .",
        "I like dog .",
    ]
    print("test data:\n", samples)
    print("my one hot encode: ")
    token_index, res = my_one_hot_encode(samples, max_len=5)
    print("my token_index:\n", token_index)
    print("my result:\n", res)
    print("keras one hot encode: ")
    keras_word_index, keras_sequences = keras_one_hot_encode(samples, max_len=5)
    print("keras_word_index:\n", keras_word_index)
    print("keras_sequences:\n", keras_sequences)


    """
    test data:
     ['I like I .', 'I like dog .']
    my one hot encode: 
    my token_index:
     {'I': 1, 'like': 2, '.': 3, 'dog': 4}
    my result:
     [[ 1.  2.  1.  3.  0.]
     [ 1.  2.  4.  3.  0.]]
    keras one hot encode: 
    keras_word_index:
     {'i': 1, 'like': 2, 'dog': 3}
    keras_sequences:
     [[1, 2, 1], [1, 2, 3]]

    """
