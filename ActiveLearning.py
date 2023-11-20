import math
import random

import matplotlib.pyplot as plt
import numpy as np


class ActiveLearning:
    selected_list = []
    selected_labels = []
    n_selected = 0

    def __init__(self, n_category, n_train, train_list, train_feature_list, labels, target_label):
        self.n_category = n_category
        self.n_train = n_train
        self.selected = np.zeros((n_category, n_train))  # metric to record the selection of samples
        self.distance_metric = np.zeros((self.n_category, self.n_train))
        self.distance_list = []
        self.benchmark = 0
        self.train_list = train_list
        self.train_feature_list = train_feature_list
        self.labels = labels
        self.target_label = target_label

    def random_sample(self, n_sample):
        self.selected_list = []
        self.selected_labels = []
        filenames = []

        # add one sample to target category to ensure that the training set contains as least one sample from target
        target_index = self.labels.index(self.target_label)
        i = random.randint(0, self.n_train - 1)
        self.selected_list.append(self.train_feature_list[target_index][i])
        self.selected_labels.append(self.target_label)
        self.selected[target_index][i] = 1
        self.n_selected += 1
        filenames.append(self.target_label + '/' + self.train_list[target_index][i])

        # randomly selected the rest 19 samples
        while self.n_selected < n_sample:
            i = random.randint(0, self.n_train * self.n_category - 1)
            row = math.floor(i / self.n_train)
            col = i % self.n_train
            if self.selected[row][col] == 0:
                self.selected[row][col] = 1
                self.n_selected += 1
                sample = self.train_feature_list[row][col]
                self.selected_list.append(sample)

                label = self.target_label if self.labels[row] == self.target_label else 'other'
                self.selected_labels.append(label)
                filenames.append(self.labels[row] + '/' + self.train_list[row][col])

        return self.selected_list, self.selected_labels, filenames

    def get_distance(self, clf):
        self.distance_list = []

        for i in range(self.n_category):
            tmp_list = clf.decision_function(self.train_feature_list[i])
            for j in range(self.n_train):
                distance = np.square(tmp_list[j])
                distance = np.sum(distance)
                self.distance_list.append(distance)
                self.distance_metric[i][j] = distance

        self.distance_list.sort()
        self.benchmark = self.distance_list[19]

        return self.distance_metric, self.benchmark

    def active_sample(self, n_sample):
        bench_id = 20
        n_selected = 0
        search_time = 0
        filenames = []

        while True:
            if search_time > 900 or n_selected == n_sample:
                break
            i = random.randint(0, self.n_train * self.n_category - 1)
            row = math.floor(i / self.n_train)
            col = i % self.n_train
            search_time += 1
            if self.distance_metric[row][col] <= self.benchmark:
                # if selected[row][col] == 0 and distance_metric[row][col] <= benchmark:
                if self.selected[row][col] == 0:
                    self.selected[row][col] = 1
                    filenames.append(self.labels[row] + '/' + self.train_list[row][col])

                    n_selected += 1
                    sample = self.train_feature_list[row][col]
                    self.selected_list.append(sample)
                    label = self.target_label if self.labels[row] == self.target_label else 'other'
                    self.selected_labels.append(label)
                else:
                    self.benchmark = self.distance_list[bench_id]
                    bench_id += 1

        return self.selected_list, self.selected_labels, filenames

    def get_result(self, clf, top_k):
        self.get_distance(clf)
        self.selected = np.zeros((self.n_category, self.n_train))
        result = []
        filenames = []

        for i in range(self.n_category):
            for j in range(self.n_train):
                if self.distance_metric[i][j] < self.benchmark:
                    self.selected[i][j] = 1
                    result.append(self.labels[i])
                    filenames.append(self.labels[i] + '/' + self.train_list[i][j])

        while len(result) < top_k:
            i = random.randint(0, self.n_train * self.n_category - 1)
            row = math.floor(i / self.n_train)
            col = i % self.n_train
            if self.selected[row][col] == 0 and self.distance_metric[row][col] <= self.benchmark:
                self.selected[row][col] = 1
                result.append(self.labels[row])
                filenames.append(self.labels[row] + '/' + self.train_list[row][col])

        return filenames

    def show_result(self, selected, train_list):
        plt.figure()
        cnt = 1
        for i in range(self.n_category):
            for j in range(self.n_train):
                if selected[i][j]:
                    selected[i][j] = 1
                    plt.subplot(4, 5, cnt)
                    cnt += 1
                    path = 'dataset/training_set/' + self.labels[i] + '/' + train_list[i][j]
                    img = plt.imread(path)
                    plt.imshow(img)
                    plt.xticks([])
                    plt.yticks([])

        plt.show()

    def show_result_file(self, filenames):
        plt.figure()
        cnt = 1
        for i in range(len(filenames)):
            path = 'dataset/training_set/' + filenames[i]
            plt.subplot(4, 5, cnt)
            cnt += 1
            img = plt.imread(path)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

        plt.show()
