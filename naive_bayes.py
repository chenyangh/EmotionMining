import pickle
from sklearn.naive_bayes import MultinomialNB
import os
import csv
import sklearn.metrics as sm
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
data_path = 'data'
data_path1 = '3100DB'




def meass(y_true, y_pred):
    m, n = y_pred.shape


class NaiveClassifier:
    def __init__(self, num):
        self.model = MultinomialNB()
        #self.model = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42, n_jobs=12)
        #self.model = RandomForestClassifier(n_estimators=300, n_jobs=12, criterion='entropy')
        self.num = num

    def gen_label(self, label):
        new_label = []
        for item in label:
            if self.num in item:
                new_label.append(1)
            else:
                new_label.append(-1)
        return new_label

    def fit(self, data, label):
        new_label = self.gen_label(label)
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
    with open('train_test_ids_4k.pkl', 'br') as f:
        train_ids, test_ids = pickle.load(f)
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    #
    bow_file = os.path.join(data_path, 'Presence_Vector.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        __train_data = []
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in train_ids:
                __train_data.append(vector)

    bow_file = os.path.join(data_path, 'Presence_Vector.csv')
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

    if True:
        print('loading data...')
        train_data, train_label, test_data, test_label = load_data()

        with open('all_data', 'bw') as f:
            pickle.dump([train_data, train_label, test_data, test_label], f)
    else:
        with open('all_data', 'br') as f:
            train_data, train_label, test_data, test_label = pickle.load(f)

    # Train model
    # list_classifiers = []
    # for i in range(num_classes):
    #     new_classifier = NaiveClassifier(i)
    #     print('Training', str(i), 'th classifier...')
    #     new_classifier.fit(train_data, train_label)
    #     list_classifiers.append(new_classifier)

    # with open('list_classifiers', 'bw') as f:
    #     pickle.dump(list_classifiers, f)
    #
    # with open('list_classifiers', 'br') as f:
    #     list_classifiers = pickle.load(f)

    # Test the classifiers

    n = len(test_label)
    # result = np.zeros([n, num_classes])
    # for i in range(num_classes):
    #     a_classifier = list_classifiers[i]
    #     result[:, i] = a_classifier.predict(test_data, threshold)
    # result = result.tolist()

    # with open('result', 'bw') as f:
    #     pickle.dump(result, f)

    # with open('result', 'br') as f:
    #     result = pickle.load(f)
    #
    # # magic start
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multioutput import MultiOutputClassifier

    train_label_b = MultiLabelBinarizer().fit_transform(train_label)
    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=20))
    clf.fit(train_data, train_label_b)
    result = clf.predict(test_data)

    import matlab.engine
    eng = matlab.engine.start_matlab()
    print(eng.F_measure(result.tolist(), test_label, n, num_classes))

    # eng.edit('average_precision.m', nargout=0)
    # eng.edit('F_measure.m', nargout=0)
    # eng.edit('test.m', nargout=0)