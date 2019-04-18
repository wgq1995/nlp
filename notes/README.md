# 对于RNN相关知识的总结
    原始HAN网络：
        对文档进行分类，输入词向量序列后，通过词级别的Bi-GRU后，每个词都会有一个对应的Bi-GRU输出的隐向量h，再通过w向量与每个时间步的h向量点积得到  
        attention权重，然后把h序列做一个根据attention权重的加权和，得到句子summary向量s2，每个句子再通过同样的Bi-GRU结构再加attention得到最终输出
        的文档特征向量v向量，然后v向量通过后级dense层再加分类器得到最终的文本分类结果。模型结构非常符合人的从词->句子->再到篇章的理解过程
    类似HAN网络思路：
        输入 --> 词嵌入 --> 双向的LSTM/GRU --> attention/globalmaxpool/globalaveragepool --> 全连接层
    Text-CNN模型
        输入 --> 词嵌入 --> CNN + MAXPOOLing --> 全连接层
    
## one hot编码([one hot encode](https://github.com/wgq1995/nlp/blob/master/notes/one_hot.py))
主要测试tokenizer的相关功能，并自己实现一个类似的功能

## 在具体任务中训练embedding([train my embedding](https://github.com/wgq1995/nlp/blob/master/notes/train_embedding.py))
只要引入Embedding层，定义任务训练即可

## 对于pad_sequences的测试([how does pad_sequences work](https://github.com/wgq1995/nlp/blob/master/notes/pad_sequences.py))
补全或者截断时，可以选择从前（pre)或者从后(post)开始

## 简单全连接层的文本分类([use pre_trained embedding but no RNN](https://github.com/wgq1995/nlp/blob/master/notes/imdb_test_no_rnn.py))
    pipeline:
    1. load data
    2. data_prepare
        2.1 tokenizer.fit_on_texts: 获取出现频率最高的max_words个单词
        2.2 tokenizer.texts_to_sequences: 将单词转变为单词对应索引
        2.3 tokenizer.word_index: 获取单词～索引对应字典，例如：{'cat':1, 'dog':2}
        2.4 pad_sequences: 将句子截取/补全到预定的长度
    3. load embeddings
        3.1 embeddings_index: 单词和对应词向量，例如：{‘cat':[0.1, 0.2, 0.1]. 'dog':['0.2', '0.5', '0.1']}
        3.2 embeddings_matrix: 词向量表，将单词按2.3中的word_index中的索引填到矩阵对应位置上去，注意，索引为0的位置是没有单词的
    4. build model and train 
## 用LSTM实现文本分类([use LSTM](https://github.com/wgq1995/nlp/blob/master/notes/LSTM_example.py))
    数据准备和全连接网络进行分类一样，后面将模型添加一个LSTM层, 可以用Bidirection将单向的变成双向的LSTM

## 实现一个简单RNN的前向传播([how does simple RNN work](https://github.com/wgq1995/nlp/blob/master/notes/myRNN.py))
    RNN 伪代码：
    state_t = 0  <== t时刻的状态（初始状态为0）
    for input_t in input_sequence:  <== 对序列元素进行遍历
        output_t = f(input_t, state_t)
        state_t = output_t  <== 当前的状态更新为当前输出
    其中： f(input_t, state_t) = activation(dot(w, input_t) + dot(u, state_t) + b)

## RNN, LSTM, GRU中的参数计算([how many params in RNN layer](https://github.com/wgq1995/nlp/blob/master/notes/understand_RNN))
* SimpleRNN: output_size × （input_size + output_size + 1)
* LSTM: 4 × (input_size + output_size + 1) × output_size
* GRU: 3 × (output_size + input_size + 1) × output_size<br>
ps: 双向的话就是单向单元的两倍参数

## callback的使用（[how to use callback](https://github.com/wgq1995/nlp/blob/master/notes/callback.py))
    使用checkpoint，earlystop方法，自定义rmse
