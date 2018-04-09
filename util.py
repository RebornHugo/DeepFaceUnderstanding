import numpy as np


def choose_sample(test_score, train_num, choose_num, feature_label):
    """
    entropy based active learning, choose the data with greatest entropy
    from test set to train set.
    :param test_score: a numpy array with predict sample and their prediction probability
    :param train_num: _TRAIN_NUM (set global in order to change its value)
    :param choose_num: _CHOOSE_NUM
    :param feature_label: a list whose element is a tuple => (file_names, labels)
    :return: (feature_label after active learning, train_num+choose_num)
    """
    d = []
    for i, score in enumerate(test_score, start=train_num):
        # 第i个样本的预测
        criterion = -sum(score * np.log(score))
        d.append((i, criterion))
    d = sorted(d, key=lambda value: value[1], reverse=True)
    samples = d[:choose_num]
    for i, temp in enumerate(samples, start=train_num):
        feature_label[i], feature_label[temp[0]] = feature_label[temp[0]], feature_label[i]
    return feature_label, train_num+choose_num
