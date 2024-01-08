import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import SpectralClustering

KEY = {
    'Windm1': {
        'real_start': 0,
        'real_end': 0,
        'nwp_start': 0,
        'nwp_end': 0,
        'real_speed_name': 'C_WS',
        'real_speed_limit': '<30',
        'real_direction_name': 'C_WD',
        'nwp_speed_name': 'C_SPEED',
        'nwp_speed_limit': '0',
        'nwp_direction_name': 'C_DIRECTION',
        'power_name': 'C_VALUE',
        'file_dir': '',
        'n_clusters': 5,
        'gamma': 10
    },

    'Windm2': {
        'real_start': 0,
        'real_end': 0,
        'nwp_start': 0,
        'nwp_end': 0,
        'real_speed_name': 'C_WS70',
        'real_speed_limit': '0',
        'real_direction_name': 'C_WD70',
        'nwp_speed_name': 'C_SPEED10',
        'nwp_speed_limit': '0',
        'nwp_direction_name': 'C_DIRECTION70',
        'power_name': 'C_VALUE',
        'file_dir': '',
        'n_clusters': 6,
        'gamma': 1
    },

    'Windm3': {
        'real_start': 0,
        'real_end': 0,
        'nwp_start': 0,
        'nwp_end': 0,
        'real_speed_name': 'C_WS70',
        'real_speed_limit': '0',
        'real_direction_name': 'C_WD70',
        'nwp_speed_name': 'C_SPEED10',
        'nwp_speed_limit': '0',
        'nwp_direction_name': 'C_DIRECTION70',
        'power_name': 'C_VALUE',
        'file_dir': '',
    },
    'Windm4': {
        'real_start': 0,
        'real_end': 0,
        'nwp_start': 0,
        'nwp_end': 0,
        'real_speed_name': '',
        'real_speed_limit': '',
        'real_direction_name': '',
        'nwp_speed_name': 'C_WS30',
        'nwp_speed_limit': '',
        'nwp_direction_name': 'C_WD30',
        'power_name': 'C_REAL_VALUE',
        'file_dir': '',
    },
    'Windm5': {
        'real_start': 0,
        'real_end': 0,
        'nwp_start': 0,
        'nwp_end': 0,
        'real_speed_name': 'C_WS',
        'real_speed_limit': '',
        'real_direction_name': 'C_WD',
        'nwp_speed_name': '',
        'nwp_speed_limit': '',
        'nwp_direction_name': '',
        'power_name': 'C_ACTIVE_POWER',
        'file_dir': '',
    },
}


@dataclass
class WindArgs:
    def __init__(self, id, nwp=True, real=False, picture=False, find_key=False):
        self.id = id
        self.nwp = nwp
        self.real = real
        self.picture = picture
        self.find_key = find_key
        self.real_start = KEY[self.id]['real_start']
        self.real_end = KEY[self.id]['real_end']
        self.nwp_start = KEY[self.id]['nwp_start']
        self.nwp_end = KEY[self.id]['nwp_end']
        self.real_speed_name = KEY[self.id]['real_speed_name']
        self.real_speed_limit = KEY[self.id]['real_speed_limit']
        self.real_direction_name = KEY[self.id]['real_direction_name']
        self.nwp_speed_name = KEY[self.id]['nwp_speed_name']
        self.nwp_speed_limit = KEY[self.id]['nwp_speed_limit']
        self.nwp_direction_name = KEY[self.id]['nwp_direction_name']
        self.power_name = KEY[self.id]['power_name']
        self.file_dir = KEY[self.id]['file_dir']
        self.n_clusters = KEY[self.id]['n_clusters']
        self.gamma = KEY[self.id]['gamma']

    def read(self):
        if self.nwp:
            nwp_data = pd.read_csv(self.file_dir)[self.nwp_start:self.nwp_end]
            return nwp_data
        elif self.real:
            real_data = pd.read_csv(self.file_dir)[self.real_start:self.real_end]
            return real_data

    def print_obj(self):
        print(self.__dict__)

    def transformer_sd(self, nwp_s, nwp_d, power):
        max_power = max(power)
        for _, (s, d) in enumerate(zip(nwp_s, nwp_d)):
            if d < 0.4:
                power[_] = power[_] / max_power
        return nwp_s, nwp_d, power

    def pre_weather(self, data):
        nwp_backnumber = ['10', '30', '50', '70', '80', '90', '110']
        for i, nwp_bn in enumerate(nwp_backnumber):
            if self.real:
                data[self.real_direction_name + nwp_bn] = data[self.real_direction_name + nwp_bn].apply(
                    lambda x: (x - 0.00001) / 360)
                if '<' in self.real_speed_limit:
                    data[self.real_speed_name + nwp_bn] = data[self.real_speed_name + nwp_bn].apply(
                        lambda x: x if x < int(self.real_speed_limit[1:]) else 0)
                elif '>' in self.real_speed_limit:
                    data[self.real_speed_name + nwp_bn] = data[self.real_speed_name + nwp_bn].apply(
                        lambda x: max(x, int(self.real_speed_limit[1:])))
            else:
                data[self.nwp_direction_name + nwp_bn] = data[self.nwp_direction_name + nwp_bn].apply(
                    lambda x: (x - 0.00001) / 360)

                if '<' in self.nwp_speed_limit:
                    data[self.nwp_speed_name + nwp_bn] = data[self.nwp_speed_name + nwp_bn].apply(
                        lambda x: min(x, int(self.nwp_speed_limit[1:])))
                elif '>' in self.nwp_speed_limit:
                    data[self.nwp_speed_name + nwp_bn] = data[self.nwp_speed_name + nwp_bn].apply(
                        lambda x: max(x, int(self.nwp_speed_limit[1:])))
            if i == 0:
                WD_Data = np.dstack((data[self.real_speed_name + nwp_bn].values if self.real else data[
                    self.nwp_speed_name + nwp_bn].values,
                                     data[self.real_direction_name + nwp_bn].values if self.real else data[
                                         self.nwp_direction_name + nwp_bn].values, data[self.power_name]))[0]
            else:
                WD_Data = np.concatenate(((np.dstack((data[self.real_speed_name + nwp_bn].values if self.real else data[
                    self.nwp_speed_name + nwp_bn].values,
                                                      data[self.real_direction_name + nwp_bn].values if self.real else
                                                      data[
                                                          self.nwp_direction_name + nwp_bn].values))[0]), WD_Data),
                                         axis=1)

        return WD_Data

    def picture_matplob(self, num, dim, category, scaled):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        color = ['r', 'g', 'k', 'c', 'y', 'm', 'b', 'cornflowerblue', 'darkgoldenrod', 'sandybrown']
        for p in range(0, num):
            y = category[p]
            ax.scatter(scaled[p, 0], scaled[p, 1], scaled[p, 2], c=color[int(y)], label='Class {}'.format(y))
        if self.real:
            plt.title('REAL Wind Speed & Wind Direction vs. Wind Turbine Power')
        else:
            plt.title('NWP Wind Speed & Wind Direction vs. Wind Turbine Power')
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Wind Direction')
        ax.set_zlabel('Power (kW)')
        ax.view_init(0, 20)
        plt.show()

    def calculate_gram(self, Normal_caled):
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

    def train_fit(self):
        data = self.read()
        power = data[self.power_name].values
        scaled = self.pre_weather(data)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        Normal_caled = min_max_scaler.fit_transform(scaled)
        if self.find_key:
            n_clusters, gamma = self.calculate_gram(Normal_caled)
        else:
            n_clusters, gamma = self.n_clusters, self.gamma
        spectral = SpectralClustering(n_clusters=n_clusters, gamma=gamma)
        category = spectral.fit_predict(Normal_caled)
        category = spectral.fit(Normal_caled)
        print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(Normal_caled, category.labels_))
        num, dim = scaled.shape
        if self.picture:
            self.picture_matplob(num, dim, category, scaled)
        return spectral, category, False


def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


if __name__ == '__main__':
    series = WindArgs('J00018')
    series.train_fit()
