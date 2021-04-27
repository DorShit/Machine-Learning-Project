import math
from sklearn.model_selection import KFold


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


class CostSensitiveID3DecisionTree:

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
            if candidate['diagnosis'] == self.labels[0]:
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
            if candidate['diagnosis'] == 'M':
                m_counter += 1
            else:
                b_counter += 0.1
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

    def CID3(self):
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
            node.value = new_default_label
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


def calculate_cost_sensitive_id3_loss(train, test, labels, m):
    id3 = CostSensitiveID3DecisionTree(get_data(train), get_features(train), m, labels)
    id3.CID3()
    test_data = get_data(test)
    loss = 0
    total_of_candidates = len(test_data)
    for candidate in test_data:
        label_to_check = id3.predict(candidate, id3.full_tree)
        if not candidate['diagnosis'] == label_to_check:
            if label_to_check == 'M' and candidate['diagnosis'] == 'B':
                loss += 0.1
            if label_to_check == 'B' and candidate['diagnosis'] == 'M':
                loss += 1
    return loss / total_of_candidates


def experiment(train, labels):
    train_data = get_data(train)
    min_loss = 1
    m_to_return = 1
    m_values = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
    for i in m_values:
        total_loss = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=311397475)
        for train_index, test_index in kf.split(train_data):
            train_group = []
            test_group = []
            for j in train_index:
                train_group.append(train_data[j])
            for k in test_index:
                test_group.append(train_data[k])
            id3 = CostSensitiveID3DecisionTree(train_group, get_features(train), i, labels)
            id3.CID3()
            loss = 0
            total_of_candidates = len(test_group)
            for candidate in test_group:
                label_to_check = id3.predict(candidate, id3.full_tree)
                if not candidate['diagnosis'] == label_to_check:
                    if label_to_check == 'M' and candidate['diagnosis'] == 'B':
                        loss += 0.1
                    if label_to_check == 'B' and candidate['diagnosis'] == 'M':
                        loss += 1
            total_loss += loss / total_of_candidates
        total_loss = total_loss / 5
        if min_loss > total_loss:
            m_to_return = i
            min_loss = total_loss
    return m_to_return


"""

The experiment will run here and get the best M. 
m_to_cal = experiment('train.csv', ['M', 'B']) 
print(m_to_cal)

"""
best_m_that_i_got_from_experiment = 2
print(calculate_cost_sensitive_id3_loss('train.csv', 'test.csv', ['M', 'B'], best_m_that_i_got_from_experiment))
















