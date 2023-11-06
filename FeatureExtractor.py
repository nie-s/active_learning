import os

import cv2
import numpy as np


# labels = ['beaches', 'bus', 'dinosaurs', 'elephants', 'flowers', 'foods', 'horses', 'monuments', 'mountains_and_snow',
#           'people_and_villages_in_Africa']  # labels of dataset


class FeatureExtractor:
    train_list = []
    test_list = []
    labels = []
    path = ''
    train_path = ''
    test_path = ''

    def load_data(self, directory_name):
        labels = []
        self.train_path = directory_name + 'training_set/'
        self.test_path = directory_name + 'test_set/'

        for _, dirnames, filenames in os.walk(self.train_path):
            if len(dirnames) > 0:
                labels = dirnames
            if len(filenames) > 0:
                self.train_list.append(filenames)

        for _, _, filenames in os.walk(self.test_path):
            if len(filenames) > 0:
                self.test_list.append(filenames)

        self.labels = labels

        return self.train_list, self.test_list, labels

    def extract_feature(self):
        sift = cv2.xfeatures2d.SIFT_create()
        train_feature_list = []
        test_feature_list = []

        for i in range(len(self.train_list)):
            img_list = self.train_list[i]
            feature_list = []

            for img_name in img_list:
                img = cv2.imread(self.train_path + self.labels[i] + "/" + img_name)
                keypoints, descriptors = sift.detectAndCompute(img, None)
                feature_list.append(descriptors[0])

            train_feature_list.append(feature_list)

        for i in range(len(self.test_list)):
            img_list = self.test_list[i]
            feature_list = []

            for img_name in img_list:
                img = cv2.imread(self.test_path + self.labels[i] + "/" + img_name)
                keypoints, descriptors = sift.detectAndCompute(img, None)
                feature_list.append(descriptors[0])

            test_feature_list.append(feature_list)

        return train_feature_list, test_feature_list

    def extract_feature_local(self, target_label):

        # store feature data just to save time :)

        # a = np.array(train_feature_list)
        #
        # with open('data.txt', 'w') as outfile:
        #     for slice_2d in train_feature_list:
        #         np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')

        train_feature_list = np.loadtxt('train.txt', delimiter=',').reshape((10, 90, 128))
        test_feature_list = np.loadtxt('test.txt', delimiter=',').reshape((10, 10, 128))

        labels = []
        for i in range(0, 10):
            tmp_list = []
            for j in range(0, 10):
                label = target_label if self.labels[i] == target_label else 'other'
                tmp_list.append(label)
            labels.append(tmp_list)

        return train_feature_list, test_feature_list, labels
