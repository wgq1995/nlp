# 对于RNN相关知识的总结
## one hot编码
[one hot encode](https://github.com/wgq1995/nlp/blob/master/notes/one_hot.py)

## 在具体任务中训练embedding
[train my embedding](https://github.com/wgq1995/nlp/blob/master/notes/train_embedding.py)

## 对于pad_sequences的测试
[how does pad_sequences work](https://github.com/wgq1995/nlp/blob/master/notes/pad_sequences.py)

## 简单全连接层的文本分类
[use pre_trained embedding but no RNN](https://github.com/wgq1995/nlp/blob/master/notes/imdb_test_no_rnn.py)

## 实现一个简单RNN的前向传播
    RNN 伪代码：
    state_t = 0  <== t时刻的状态（初始状态为0）
    for input_t in input_sequence:  <== 对序列元素进行遍历
        output_t = f(input_t, state_t)
        state_t = output_t  <== 当前的状态更新为当前输出
    其中： f(input_t, state_t) = activation(dot(w, input_t) + dot(u, state_t) + b)
[how does simple RNN work](https://github.com/wgq1995/nlp/blob/master/notes/myRNN.py)

## RNN, LSTM, GRU中的参数计算
[how many params in RNN layer](https://github.com/wgq1995/nlp/blob/master/notes/understand_RNN)
