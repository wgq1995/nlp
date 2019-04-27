from sklearn.model_selection import KFold
"""
注意点：对于不能均等份的数据集，其前n_samples % n_splits子集拥有n_samples // n_splits + 1个样本，
其余子集都只有n_samples // n_splits样本
"""
a = list(range(10))
b = [0] * 5 + [1] * 5
kf = KFold(n_splits=3, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(a, b)):
    print("split {}:".format(i))
    print("train_index: {}".format(train_index))
    print("test_index: {}".format(test_index))
