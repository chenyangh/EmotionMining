import pickle
from sklearn.naive_bayes import MultinomialNB
import os
import csv
import sklearn.metrics as sm
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from measurements import CalculateFM
from random import shuffle

def meass(y_true, y_pred):
    m, n = y_pred.shape


class OneClassifier:
    def __init__(self, num):
        self.model = MultinomialNB()
        #self.model = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42, n_jobs=12)
        # self.model = RandomForestClassifier(n_estimators=300, n_jobs=12, criterion='entropy')
        self.num = num

    def gen_label(self, label):
        new_label = []
        for item in label:
            if self.num in item:
                new_label.append(1)
            else:
                new_label.append(-1)
        return new_label

    def gen_balanced_label(self, label):
        new_label = []
        for item in label:
            if self.num in item:
                new_label.append(1)
            else:
                new_label.append(-1)
        return new_label


    def fit(self, data, label, is_balanced=False):
        if not is_balanced:
            new_label = self.gen_label(label)
            self.model.fit(data, new_label)
        else:
            new_label = self.gen_label(label)
            pos_idx = [idx for idx, d in enumerate(new_label) if d == 1]
            neg_idx = [idx for idx, d in enumerate(new_label) if d != 1]
            shuffle(neg_idx)
            neg_idx = neg_idx[:len(pos_idx)]
            new_label = np.array(new_label)
            new_idx = np.concatenate((pos_idx, neg_idx))

            data = data[new_idx, :]
            new_label = new_label[new_idx]
            self.model.fit(data, new_label)


    def predict(self, data,  __threshold):
        pred_result = self.model.predict(data)
        # result_with_threshold = []
        # for item in pred_result:
        #     if item[0] > __threshold:
        #         result_with_threshold.append(1)
        #     else:
        #         result_with_threshold.append(-1)
        return pred_result

def load_data():
    # Load the data
    data_path = 'data'

    with open('others/train_test_ids_31k.pkl', 'br') as f:
        train_ids, test_ids = pickle.load(f)
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    #
    bow_file = os.path.join(data_path, 'Presence_Vector_31k_train.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        __train_data = []
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in train_ids:
                __train_data.append(vector)

    bow_file = os.path.join(data_path, 'Presence_Vector_31k_test.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        __test_data = []
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in test_ids:
                __test_data.append(vector)

    # read integer labels for train, but binary labels for test
    bow_file = os.path.join(data_path, 'integer_labels.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        __train_label = []
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in train_ids:
                __train_label.append(vector)

    bow_file = os.path.join(data_path, 'bipolar_labels.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        __test_label = []
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in test_ids:
                __test_label.append(vector)

    return __train_data, __train_label, __test_data, __test_label

# train 9 naives bayes classifier
if __name__ == '__main__':
    threshold = 0.8
    num_classes = 9

    print('loading data...')
    if False:
        train_data, train_label, test_data, test_label = load_data()

        with open('data/all_data', 'bw') as f:
            pickle.dump([train_data, train_label, test_data, test_label], f)
    else:
        with open('data/all_data', 'br') as f:
            train_data, train_label, test_data, test_label = pickle.load(f)
    n = len(test_label)
    print('Finish loading data...')



    # eng.edit('average_precision.m', nargout=0)
    # eng.edit('F_measure.m', nargout=0)
    # eng.edit('test.m', nargout=0)

    train_data = np.array(train_data)
    # ##Train model
    list_classifiers = []
    for i in range(num_classes):
        new_classifier = OneClassifier(i)
        print('Training', str(i), 'th classifier...')
        new_classifier.fit(train_data, train_label, is_balanced=True)
        list_classifiers.append(new_classifier)

    # with open('list_classifiers', 'bw') as f:
    #     pickle.dump(list_classifiers, f)
    #
    # with open('list_classifiers', 'br') as f:
    #     list_classifiers = pickle.load(f)

    ## Test the classifiers

    result = np.zeros([n, num_classes])
    for i in range(num_classes):
        a_classifier = list_classifiers[i]
        result[:, i] = a_classifier.predict(test_data, threshold)
    result = result.tolist()

    results = CalculateFM(test_label, result)

    print('Results:', results)
    # with open('result', 'bw') as f:
    #     pickle.dump(result, f)

    # with open('result', 'br') as f:
    #     result = pickle.load(f)
    #

    # # magic start

    #  ##  One vs rest wrapper
    # from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    # from sklearn.preprocessing import MultiLabelBinarizer
    # from sklearn.multioutput import MultiOutputClassifier
    #
    # train_label_b = MultiLabelBinarizer().fit_transform(train_label)
    # clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=20))
    # clf.fit(train_data, train_label_b)
    # result = clf.predict(test_data)
    #
    # import matlab.engine
    #
    # eng = matlab.engine.start_matlab()
    # print(eng.F_measure(result.tolist(), test_label, n, num_classes))