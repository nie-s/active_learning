import os

from sklearn.svm import SVC

from ActiveLearning import ActiveLearning
from FeatureExtractor import FeatureExtractor

n_train = 90  # size of one category in training set
n_test = 10  # size of one category in test set
n_category = 10

path = os.getcwd() + '/dataset/'
print("input:")
# target_label = input()
target_label = 'bus'

extractor = FeatureExtractor()
train_list, test_list, labels = extractor.load_data(path)
# train_feature_list, test_feature_list, test_labels = extractor.extract_feature(target_label)
train_feature_list, test_feature_list, test_labels = extractor.extract_feature_local(target_label)

active_learning = ActiveLearning(n_category, n_train, train_list, train_feature_list, labels, target_label)
clf = SVC(gamma='auto')
# clf = SVC(gamma='auto', kernel='poly')

# first, randomly select 20 samples to get start
selected_list, selected_labels, filenames = active_learning.random_sample(20)
active_learning.show_result_file(filenames)

clf.fit(selected_list, selected_labels)


# then, actively train the svm by selecting 20 samples that closet to the hyperplane
for i in range(1, 5):
    print('ROUND ' + str(i))

    distance_metric, benchmark = active_learning.get_distance(clf)
    selected_list, selected_labels, filenames = active_learning.active_sample(20)
    active_learning.show_result_file(filenames)
    clf.fit(selected_list, selected_labels)

# print(selected_labels)

# finally, get the result
top_k = 20
result = active_learning.get_result(clf, top_k)
active_learning.show_result(active_learning.selected, train_list)
# print(result)


# print(clf.predict(np.reshape(test_feature_list, (-1, 144))))
