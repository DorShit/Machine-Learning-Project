import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd


class Node:
    def __init__(self):
        self.value = ''
        self.t_value = None
        self.left_child = None
        self.right_child = None


def get_data(file_name):
    data_list = []
    with open(file_name) as f:
        for line in map(str.strip, f):
            data_list.append(line.split(','))
    headline = data_list.pop(0)

    data_final_list = []
    for item in data_list:
        data_dict = {}
        for i in range(len(item)):
            data_dict[headline[i]] = item[i]
        data_final_list.append(data_dict)
    return data_final_list


def get_features(file_name):
    data_list = []
    with open(file_name) as f:
        for line in map(str.strip, f):
            data_list.append(line.split(','))
    features = data_list.pop(0)
    features.pop(0)
    return features


class ID3DecisionTree:

    def __init__(self, data, features, m, labels):
        self.data = data
        self.features = features
        self.labels = labels
        self.full_tree = None
        self.m = m
        self.default_label = self.get_majority_label(data)
        self.entropy = self.get_entropy(self.data)

    def get_majority_label(self, example_group):
        m_counter = 0
        b_counter = 0
        for candidate in example_group:
            if candidate['diagnosis'] == 'M':
                m_counter += 1
            else:
                b_counter += 1
        if b_counter < m_counter:
            return 'M'
        return 'B'

    def get_entropy(self, example_group):
        m_counter = 0
        b_counter = 0
        for candidate in example_group:
            if candidate['diagnosis'] == self.labels[0]:
                m_counter += 1
            else:
                b_counter += 1
        label_counter = [m_counter, b_counter]
        entropy = 0
        for counter in label_counter:
            if counter != 0:
                entropy += -counter/len(example_group) * math.log(counter/len(example_group), 2)

        return entropy

    def information_gain(self, example_group, feature_name, t_values):
        total_entropy = self.get_entropy(example_group)
        max_information_gain = 0
        max_t = 0
        for t_value in t_values:
            features_value_of_group_0 = []
            features_value_of_group_1 = []
            feature_value = [features_value_of_group_0, features_value_of_group_1]
            features_group_counter = []
            information = 0
            information_gain = 0
            for candidate in example_group:
                if float(candidate[feature_name]) < t_value:
                    features_value_of_group_0.append(candidate)
                else:
                    features_value_of_group_1.append(candidate)
            features_group_counter.append(len(features_value_of_group_0))
            features_group_counter.append(len(features_value_of_group_1))
            for i in range(len(features_group_counter)):
                information += (features_group_counter[i] / len(example_group)) * self.get_entropy(feature_value[i])
            information_gain = total_entropy - information
            if max_information_gain <= information_gain:
                max_information_gain = information_gain
                max_t = t_value

        return max_information_gain, max_t

    def max_info_gain(self, example_group):
        best_t = {}
        feature_to_return = ''
        feature_value_to_choose = 0
        for feature in self.features:
            feature_values = []
            t_values = []
            for candidate in example_group:
                feature_values.append(float(candidate[feature]))
            feature_values.sort()
            feature_values = list(dict.fromkeys(feature_values)) # getting rid of duplication
            for i in range(len(feature_values) - 1):
                t_values.append((float(feature_values[i]) + float(feature_values[i + 1])) / 2)
            entropy_to_choose, best_t[feature] = self.information_gain(example_group, feature, set(t_values))
            if feature_value_to_choose <= entropy_to_choose:
                feature_value_to_choose = entropy_to_choose
                feature_to_return = feature
        t_to_return = best_t.get(feature_to_return)
        return feature_to_return, t_to_return

    def ID3(self):
        self.full_tree = self.fit(self.data, self.default_label, self.features)

    def fit(self, example_group, default_label, features):
        node = Node()
        labels = []
        new_default_label = self.get_majority_label(example_group)

        if len(example_group) < self.m:
            node.value = default_label
            return node

        for candidate in example_group:
            labels.append(candidate['diagnosis'])
        if len(set(labels)) == 1:  # Only 1 label -> a leaf
            node.value = labels[0]
            return node

        if len(features) == 0:  # no more features
            node.value = new_default_label
            return node

        max_feature, max_t = self.max_info_gain(example_group)
        node.value = max_feature
        node.t_value = max_t
        greater_than_t = []
        lesser_than_t = []

        for candidate in example_group:
            if float(candidate[max_feature]) < max_t:
                lesser_than_t.append(candidate)
            else:
                greater_than_t.append(candidate)

        node.left_child = self.fit(lesser_than_t, new_default_label, features)
        node.right_child = self.fit(greater_than_t, new_default_label, features)

        return node

    def predict(self, candidate, node):
        if node.value in self.labels:
            return node.value
        else:
            if float(candidate[node.value]) < node.t_value:
                return self.predict(candidate, node.left_child)
            else:
                return self.predict(candidate, node.right_child)






def calculate_id3_precision(train, test, labels):
    id3 = ID3DecisionTree(get_data(train), get_features(train), 1, labels)
    id3.ID3()
    test_data = get_data(test)
    total_of_candidates = len(test_data)
    total_of_right_decisions = 0
    for candidate in test_data:
        if candidate['diagnosis'] == id3.predict(candidate, id3.full_tree):
            total_of_right_decisions += 1
    return total_of_right_decisions / total_of_candidates


def calculate_id3_loss(train, test, labels):
    id3 = ID3DecisionTree(get_data(train), get_features(train), 1, labels)
    id3.ID3()
    test_data = get_data(test)
    total_of_candidates = len(test_data)
    loss = 0
    for candidate in test_data:
        label_to_check = id3.predict(candidate, id3.full_tree)
        if not candidate['diagnosis'] == label_to_check:
            if label_to_check == 'M' and candidate['diagnosis'] == 'B':
                loss += 0.1
            if label_to_check == 'B' and candidate['diagnosis'] == 'M':
                loss += 1

    return loss / total_of_candidates


"""
3.3 
TODO: run the experiment function to get the graph : experiment('train.csv', ['M', 'B'])
"""


def experiment(train, labels):
    train_data = get_data(train)
    precision_per_m = []
    m_values = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
    for i in m_values:
        average_acc = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=311397475)
        for train_index, test_index in kf.split(train_data):
            train_group = []
            test_group = []
            for j in train_index:
                train_group.append(train_data[j])
            for k in test_index:
                test_group.append(train_data[k])
            id3 = ID3DecisionTree(train_group, get_features(train), i, labels)
            id3.ID3()
            total_of_candidates = len(test_group)
            total_of_right_decisions = 0
            for candidate in test_group:
                if candidate['diagnosis'] == id3.predict(candidate, id3.full_tree):
                    total_of_right_decisions += 1
            average_acc += total_of_right_decisions / total_of_candidates
        precision_per_m.append(average_acc / 5)
    make_graph(precision_per_m, m_values)



def make_graph(precisions, m_values):
    fig, ax1 = plt.subplots()
    np_precision = np.array(precisions)
    np_m_values = np.array(m_values)
    p1, = ax1.plot(np_m_values, np_precision, color='red')
    ax1.set_ylabel('Precision', color='black')
    ax1.tick_params('y', colors='black')
    ax1.set_xlabel('M')
    fig.tight_layout()
    plt.title(f'Pruning effect on Precision at k_folds')
    plt.show()


def main():
    print(calculate_id3_precision('train.csv', 'test.csv', ['M', 'B']))
    """
    experiment('train.csv', ['M', 'B']) --> For Graph 
    print(calculate_id3_loss('train.csv', 'test.csv', ['M', 'B'])) --> For loss 
    """


if __name__ == '__main__':
    main()











