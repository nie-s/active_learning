import os

import cv2
import numpy as np
# labels = ['beaches', 'bus', 'dinosaurs', 'elephants', 'flowers', 'foods', 'horses', 'monuments', 'mountains_and_snow',
#           'people_and_villages_in_Africa']  # labels of dataset
import pywt


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

    def color_feature(self, image):
        color_ranges = [
            # 红色
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (180, 255, 255)),
            # 橙色
            ((11, 100, 100), (18, 255, 255)),
            # 黄色
            ((19, 100, 100), (30, 255, 255)),
            # 绿色
            ((31, 100, 100), (70, 255, 255)),
            # 青色/蓝绿色
            ((71, 100, 100), (90, 255, 255)),
            # 蓝色
            ((91, 100, 100), (126, 255, 255)),
            # 紫色
            ((127, 100, 100), (150, 255, 255)),
            # 粉色
            ((151, 100, 100), (169, 255, 255)),
            # 棕色
            ((10, 100, 20), (20, 255, 200)),
            # 黑色
            ((0, 0, 0), (180, 255, 50)),
            # 白色和灰色
            ((0, 0, 50), (180, 50, 255)),
        ]

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 初始化特征向量
        feature_vector = []

        # 遍历每个颜色范围，计算特征
        for lower, upper in color_ranges:
            # 创建颜色掩模
            mask = cv2.inRange(hsv_image, lower, upper)

            # 颜色掩模位
            mask_bit = 1 if np.any(mask) else 0

            # 如果掩模中有像素，计算额外的特征
            if mask_bit:
                masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
                h, s, v = cv2.split(masked_image)
                h = h[mask > 0]
                s = s[mask > 0]
                v = v[mask > 0]

                # 颜色像素数量
                color_pixels = np.count_nonzero(mask)

                # 颜色均值（HSV）
                mean_h = np.mean(h) if h.any() else 0
                mean_s = np.mean(s) if s.any() else 0
                mean_v = np.mean(v) if v.any() else 0

                # 颜色标准差（HSV）
                std_h = np.std(h) if h.any() else 0
                std_s = np.std(s) if s.any() else 0
                std_v = np.std(v) if v.any() else 0

                # 形状特征（伸长和伸展）
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    area = cv2.contourArea(largest_contour)
                    extent = area / (w * h) if w * h > 0 else 0
                else:
                    elongation = 0
                    extent = 0

                # 将掩模位和计算的特征添加到特征向量
                features = [color_pixels, mean_h, mean_s, mean_v, std_h, std_s, std_v, elongation, extent]
            else:
                # 如果没有掩模像素，使用零填充特征
                features = [mask_bit] + [0] * 8  # 一个掩模位和8个特征填充为零

            feature_vector.extend(features)
        # if len(feature_vector) != 120:
        # raise ValueError(f"Feature vector length mismatch. Expected 120, but got {len(feature_vector)}")

        return np.array(feature_vector)

    # 函数用于执行DWT，并获得三个方向的子图以及缩小的图像
    def dwt_2d_image(self, image, wavelet='haar'):
        coeffs = pywt.dwt2(image, wavelet=wavelet)
        cA, (cH, cV, cD) = coeffs
        return cA, cH, cV, cD

    # 函数用于进一步分解低频子图，并进行纹理分析
    def wavelet_packet_decomposition(self, image, wavelet='haar', max_level=3):
        # 初始化小波包树
        wp = pywt.WaveletPacket2D(data=image, wavelet=wavelet, mode='symmetric')
        subimages = []
        # 获取三个尺度和三个方向的子图
        for level in range(1, max_level + 1):
            for node in ['h', 'v', 'd']:
                subimage = wp[node * level].data
                subimages.append(subimage)
        return subimages

    def sub_texture_features(self, subimage):
        rows, cols = subimage.shape
        x_grid, y_grid = np.meshgrid(np.arange(cols), np.arange(rows))
        centroid_x = np.sum(x_grid * subimage) / np.sum(subimage)
        centroid_y = np.sum(y_grid * subimage) / np.sum(subimage)
        inertia = np.sum(((x_grid - centroid_x) ** 2 + (y_grid - centroid_y) ** 2) * subimage)
        if np.sum(subimage) != 0 and inertia >= 0:
            elongation = np.sqrt(inertia)
        else:
            elongation = 0

        # 计算分布度，这里使用标准差
        spreadness = np.std(subimage)

        return elongation, spreadness

    def texture_feature(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        cA, cH, cV, cD = self.dwt_2d_image(image)
        subimages = self.wavelet_packet_decomposition(cA, max_level=3)

        # 计算每个子图的纹理特征和空间信息
        texture_features = []
        for subimage in subimages:
            elongation, spreadness = self.sub_texture_features(subimage)
            spatial_info = np.array([subimage.shape[0], subimage.shape[1]])  # 空间信息为子图的尺寸
            texture_feature = np.concatenate((np.array([elongation, spreadness]), spatial_info))
            texture_features.append(texture_feature)

        # 将纹理特征列表转换为36维向量
        texture_vector = np.concatenate(texture_features)
        return texture_vector

    def extract_feature(self, target_label):
        train_feature_list = []
        test_feature_list = []

        for i in range(len(self.train_list)):
            img_list = self.train_list[i]
            feature_list = []

            for img_name in img_list:
                path = self.train_path + self.labels[i] + "/" + img_name
                img = cv2.imread(self.train_path + self.labels[i] + "/" + img_name)
                feature = np.concatenate((self.color_feature(img), (self.texture_feature(path))), axis=0)
                feature_list.append(feature)

            train_feature_list.append(feature_list)

        for i in range(len(self.test_list)):
            img_list = self.test_list[i]
            feature_list = []

            for img_name in img_list:
                path = self.test_path + self.labels[i] + "/" + img_name
                img = cv2.imread(path)
                feature = np.concatenate((self.color_feature(img), (self.texture_feature(path))), axis=0)
                feature_list.append(feature)

            test_feature_list.append(feature_list)

        labels = []
        for i in range(0, 10):
            tmp_list = []
            for j in range(0, 10):
                label = target_label if self.labels[i] == target_label else 'other'
                tmp_list.append(label)
            labels.append(tmp_list)

        a = np.array(train_feature_list)

        with open('train.txt', 'w') as outfile:
            for slice_2d in a:
                np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')

        return train_feature_list, test_feature_list, labels

    def extract_feature_local(self, target_label):

        # store feature data just to save time :)

        # a = np.array(train_feature_list)
        #
        # with open('data.txt', 'w') as outfile:
        #     for slice_2d in train_feature_list:
        #         np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')

        train_feature_list = np.loadtxt('train.txt', delimiter=',').reshape((10, 90, 144))
        test_feature_list = np.loadtxt('test.txt', delimiter=',').reshape((10, 10, 144))

        labels = []
        for i in range(0, 10):
            tmp_list = []
            for j in range(0, 10):
                label = target_label if self.labels[i] == target_label else 'other'
                tmp_list.append(label)
            labels.append(tmp_list)

        return train_feature_list, test_feature_list, labels
