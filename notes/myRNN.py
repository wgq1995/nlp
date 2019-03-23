"""
RNN 伪代码：
state_t = 0  <== t时刻的状态（初始状态为0）
for input_t in input_sequence:  <== 对序列元素进行遍历
    output_t = f(input_t, state_t)
    state_t = output_t  <== 当前的状态更新为当前输出

其中： f(input_t, state_t) = activation(dot(w, input_t) + dot(u, state_t) + b)
"""
import numpy as np

timesteps = 20
input_features = 5
output_features = 10

input_data = np.random.random(size=(timesteps, input_features))
state_t = np.zeros(shape=(output_features,))

w = np.random.random(size=(output_features, input_features))
u = np.random.random(size=(output_features, output_features))
b = np.random.random(size=(output_features, ))

successive_outputs = []
for input_t in input_data:
    output_t = np.tanh(np.dot(w, input_t) + np.dot(u, state_t) + b)
    state_t = output_t
    successive_outputs.append(output_t)

print(np.array(successive_outputs))
