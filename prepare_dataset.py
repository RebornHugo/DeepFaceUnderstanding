import os
import random
import pickle
import pandas as pd

data_dir = '/home/hugo/datasets/celebA'


def get_feature_label(attr, end=10000):
    images = os.path.join(data_dir, 'Anno', 'list_attr_celeba.txt')
    assert os.path.exists(images), 'not images found 233333'
    df = pd.read_csv(images, delim_whitespace=True, skip_blank_lines=1, header=1, nrows=end)
    attr = df.loc[:, attr].tolist()
    index = list(df.index.values)
    return zip(index, attr)


if __name__ == '__main__':
    train = list(get_feature_label('Young', end=10000))
