import os
import math
import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

KEY = {
    'ETTm1': {
        'value_name': 'OT',
        'file_dir': '',
        'n_clusters': 3,
        'gamma': 0.01,
        'find_key': False,
    },
    'ETTm2': {
        'value_name': 'OT',
        'file_dir': '',
        'n_clusters': 3,
        'gamma': 0.01,
        'find_key': False,
    },
    'electricity': {
        'value_name': 'OT',
        'file_dir': '',
        'n_clusters': 3,
        'gamma': 0.1,
        'find_key': False,
    },
    'traffic': {
        'value_name': 'OT',
        'file_dir': '',
        'n_clusters': 4,
        'gamma': 10,
        'find_key': False,
    },
    'weather': {
        'value_name': 'OT',
        'file_dir': '',
        'n_clusters': 3,
        'gamma': 0.1,
        'find_key': False,
    },
}


@dataclass
class Args:
    def __init__(self, id):
        self.id = id
        self.value_name = KEY[self.id]['value_name']
        self.file_dir = KEY[self.id]['file_dir']
        self.find_key = KEY[self.id]['find_key']
        self.n_clusters = KEY[self.id]['n_clusters']
        self.gamma = KEY[self.id]['gamma']

    def read(self):
        return pd.read_csv(self.file_dir)

    def print_obj(self):
        "打印对象的所有属性"
        print(self.__dict__)

    def transformer_sd(self, nwp_s, nwp_d, power):
        max_power = max(power)
        for _, (s, d) in enumerate(zip(nwp_s, nwp_d)):
            if d < 0.4:
                power[_] = power[_] / max_power

        return nwp_s, nwp_d, power

    def calculate_gram(self, Normal_caled):
        # harabasz_score 计算
        print('计算最佳分类数目与gamma值')
        scores = []
        s = dict()
        for index, gamma in enumerate((0.01, 0.1, 1, 10)):
            for index, k in enumerate((3, 4, 5, 6, 7, 8)):
                y_prediction = SpectralClustering(n_clusters=k).fit_predict(Normal_caled)

                print("Calinski-Harabasz Score with gamma=", gamma, "n_cluster=", k, "score=",
                      metrics.calinski_harabasz_score(Normal_caled, y_prediction))
                tmp = dict()
                tmp['gamma'] = gamma
                tmp['n_cluster'] = k
                tmp['score'] = metrics.calinski_harabasz_score(Normal_caled, y_prediction)
                s[metrics.calinski_harabasz_score(Normal_caled, y_prediction)] = tmp
                scores.append(metrics.calinski_harabasz_score(Normal_caled, y_prediction))
        print(np.max(scores))
        print("最大得分项：")
        print(s.get(np.max(scores)))
        return s.get(np.max(scores))['n_cluster'], s.get(np.max(scores))['gamma']

    def save_pkl(self, class_example, save_dir, save_name):
        with open(os.path.join(save_dir, save_name), 'wb') as output_hal:
            str = pickle.dumps(class_example)
            output_hal.write(str)
        print('{} saved'.format(save_name))
        return 0

    def load_pkl(self, class_example, save_dir, class_name):
        load_pkl = class_example
        with open(os.path.join(save_dir, class_name), 'rb') as file:
            load_file = pickle.loads(file.read())
        return load_pkl

    def save_ndarry(self, ndarry, save_dir, save_name):
        np.save(os.path.join(save_dir, save_name), ndarry)
        return 0

    def load_ndarry(self, save_dir, ndarry_name):
        ndarry = np.load('{}.npy'.format(os.path.join(save_dir, ndarry_name)))
        return ndarry

    def train_fit(self):
        data = self.read()
        power = data[self.value_name].values
        scaled = data.values[:, 1:]

        # 数据标准化
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

        Normal_caled = min_max_scaler.fit_transform(scaled)

        if self.find_key:
            n_clusters, gamma = self.calculate_gram(Normal_caled)
        else:
            n_clusters, gamma = self.n_clusters, self.gamma

        spectral = SpectralClustering(n_clusters=n_clusters, gamma=gamma)

        """
        spectral.fit enable
        """
        category = spectral.fit(Normal_caled)

        """
        pkl save and load
        """
        self.save_pkl(category, './', self.id)
        category = self.load_pkl(spectral, './', self.id)

        """
        npy save and load
        """
        self.save_ndarry(category.centure_, './', self.id)

        try:
            centure_ = self.load_ndarry('', self.id)
            Centure = True
            print('load cluster centure')
            return spectral, centure_, Centure
        except:
            category = spectral.fit(Normal_caled)
            Centure = False
            print('calculate cluster centure')
            return spectral, category, Centure


def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


if __name__ == '__main__':
    series = Args('ETTm1')
    series.train_fit()
