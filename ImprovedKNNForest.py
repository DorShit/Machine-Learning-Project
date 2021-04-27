from ID3 import ID3DecisionTree
from ID3 import get_features
from ID3 import get_data
import numpy as np
import random
import math
from sklearn.model_selection import KFold


class KNNDecisionTree:

    def __init__(self, n, k, data, features, labels, p):
        self.n = n
        self.k = k
        self.decision_forest = []
        self.centroid = []
        self.data = data
        self.features = features
        self.labels = labels
        self.weight_for_tree = []
        self.p = p
        self.m_best_f = int(len(self.features) / 5)

    def get_most_common_label(self, predictions):
        m_counter = 0
        b_counter = 0
        for label in predictions:
            if label[0] == 'M':
                m_counter += 1 * label[1]
            else:
                b_counter += 1 * label[1]
        if b_counter <= m_counter:
            return 'M'
        return 'B'

    def get_the_m_best_features(self):
        features_ig = {}
        features_ig_list = []
        m_best_features = []
        dummy_id3 = ID3DecisionTree(self.data, self.features, 1, self.labels)
        for feature in self.features:
            feature_values = []
            t_values = []
            for candidate in self.data:
                feature_values.append(float(candidate[feature]))
            feature_values.sort()
            feature_values = list(dict.fromkeys(feature_values))  # getting rid of duplication
            for i in range(len(feature_values) - 1):
                t_values.append((float(feature_values[i]) + float(feature_values[i + 1])) / 2)
            features_ig[feature], dummy = dummy_id3.information_gain(self.data, feature, set(t_values))
            features_ig_list.append(features_ig[feature])
        features_ig_list.sort(reverse=True)
        features_ig_list = list(dict.fromkeys(features_ig_list))
        for i in range(len(features_ig_list) - self.m_best_f):
            features_ig_list.pop(-1)
        key_list = list(features_ig.keys())
        val_list = list(features_ig.values())
        for ig in features_ig_list:
            pos = val_list.index(ig)
            m_best_features.append(key_list[pos])
        self.features = m_best_features

    def calc_m_value(self):
        precision_per_m = {}
        copy_of_data = self.data.copy()
        copy_of_features = self.features.copy()
        fea_len = len(self.features)
        m_values = range(int(fea_len / 10), int(fea_len / 3) + 1)
        for i in m_values:
            self.features = copy_of_features.copy()
            self.m_best_f = i
            self.get_the_m_best_features()
            average_acc = 0
            kf = KFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(copy_of_data):
                train_group = []
                test_group = []
                self.decision_forest = []
                self.centroid = []
                for j in train_index:
                    train_group.append(copy_of_data[j])
                for k in test_index:
                    test_group.append(copy_of_data[k])
                self.data = train_group
                self.fill_forest()
                total_of_candidates = len(test_group)
                total_of_right_decisions = 0
                for candidate in test_group:
                    if candidate['diagnosis'] == self.predict_k_trees(candidate, self.k):
                        total_of_right_decisions += 1
                average_acc += total_of_right_decisions / total_of_candidates
            precision_per_m[i] = (average_acc / 5)
            #print(precision_per_m[i], i)
        self.data = copy_of_data
        self.centroid = []
        self.decision_forest = []
        self.features = copy_of_features.copy()
        self.m_best_f = max(precision_per_m, key=precision_per_m.get)
        #print("And the Best M goes to..", self.m_best_f)

    def fill_forest(self):
        for i in range(self.n):
            data_to_tree = random.sample(self.data, int(self.p*len(self.data)))
            self.centroid.append(self.centroid_calc(data_to_tree))
            tree = ID3DecisionTree(data_to_tree, self.features, 1, self.labels)
            tree.ID3()
            self.decision_forest.append(tree)

    def centroid_calc(self, example_group):
        tree_centroid = {}
        for feature in self.features:
            avg_feature_val = 0
            if feature != 'diagnosis':
                for candidate in example_group:
                    avg_feature_val += float(candidate[feature])
            tree_centroid[feature] = avg_feature_val / len(example_group)
        return tree_centroid

    def KNN(self):
        """"
        self.calc_m_value(). Did it at the experiment for best M features. Now I set it when creating the class
        """
        self.get_the_m_best_features()
        self.fill_forest()

    def choose_k_trees(self, candidate):
        forest_dis_from_candidate = []
        for tree_centroid in self.centroid:
            tree_dis_from_candidate = 0
            for feature in self.features:
                tree_dis_from_candidate += (tree_centroid[feature] - float(candidate[feature])) ** 2
            forest_dis_from_candidate.append(math.sqrt(tree_dis_from_candidate))
        np_forest = np.array(forest_dis_from_candidate)
        if self.k < self.n:
            best_k_temp = np.argpartition(np_forest, self.k)
        else:
            best_k_temp = range(0, self.n)
        best_k = []
        for i in range(self.k):
            best_k.append(best_k_temp[i])
        k_tree_to_return = []
        for i in best_k:
            k_tree_to_return.append(self.decision_forest[i])
        return k_tree_to_return

    def predict_k_trees(self, candidate, k):
        self.k = k
        weight = k
        self.weight_for_tree = []
        for i in range(k):
            self.weight_for_tree.append(weight)
            weight -= 1
        k_trees = self.choose_k_trees(candidate)
        k_predictions = []
        j = 0
        for tree in k_trees:
            k_predictions.append([tree.predict(candidate, tree.full_tree), self.weight_for_tree[j]])
            j += 1
        return self.get_most_common_label(k_predictions)


def calculate_KNN_precision(k, test, knn):
    test_data = get_data(test)
    total_of_candidates_to_check = len(test_data)
    total_of_right_decisions = 0
    for candidate in test_data:
        prediction = knn.predict_k_trees(candidate, k)
        if candidate['diagnosis'] == prediction:
            total_of_right_decisions += 1
    return total_of_right_decisions / total_of_candidates_to_check


def get_knn_precision(train, test, n, k, p, labels):
    knn = KNNDecisionTree(n, k, get_data(train), get_features(train), labels, p)
    knn.KNN()
    return calculate_KNN_precision(k, test, knn)


best_n_from_experiment = 20
best_k_from_experiment = 11
best_p_from_experiment = 0.4
print(get_knn_precision('train.csv', 'test.csv', best_n_from_experiment, best_k_from_experiment, best_p_from_experiment, ['M', 'B']))


