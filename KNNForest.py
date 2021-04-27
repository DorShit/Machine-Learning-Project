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
        self.p = p

    def get_most_common_label(self, predictions):
        m_counter = 0
        b_counter = 0
        for label in predictions:
            if label == 'M':
                m_counter += 1
            else:
                b_counter += 1
        if b_counter <= m_counter:
            return 'M'
        return 'B'

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
        k_trees = self.choose_k_trees(candidate)
        k_predictions = []
        for tree in k_trees:
            k_predictions.append(tree.predict(candidate, tree.full_tree))
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


def experiment_for_best_n_k_p(n_values, train, labels):
    train_data = get_data(train)
    max_precision = 0
    best_n_k_p = [1, 1, 0.3]
    for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70]:
        for n in n_values:
            for k in range(1, n+1):
                average_acc = 0
                kf = KFold(n_splits=5, shuffle=True, random_state=311397475)
                for train_index, test_index in kf.split(train_data):
                    train_group = []
                    test_group = []
                    for i in train_index:
                        train_group.append(train_data[i])
                    for j in test_index:
                        test_group.append(train_data[j])
                    knn = KNNDecisionTree(n, k, train_group, get_features(train), labels, p)
                    knn.KNN()
                    total_of_candidates = len(test_group)
                    total_of_right_decisions = 0
                    for candidate in test_group:
                        if candidate['diagnosis'] == knn.predict_k_trees(candidate, k):
                            total_of_right_decisions += 1
                    average_acc += total_of_right_decisions / total_of_candidates
                    average_acc = (average_acc / 5)
                if max_precision < average_acc:
                    best_n_k_p = [n, k, p]
                    max_precision = average_acc
    return best_n_k_p




def get_knn_precision(train, test, labels):
    """
    n_k_p = experiment_for_best_n_k(n_values, train, labels) --> Get the values from the experiment
    """
    best_n_from_experiment = 20
    best_k_from_experiment = 11
    best_p_from_experiment = 0.4
    n_k_p = [best_n_from_experiment, best_k_from_experiment, best_p_from_experiment]
    knn = KNNDecisionTree(n_k_p[0], n_k_p[1], get_data(train), get_features(train), labels, n_k_p[2])
    knn.KNN()
    return calculate_KNN_precision(n_k_p[1], test, knn)


print(get_knn_precision('train.csv', 'test.csv', ['M', 'B']))


