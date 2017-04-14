import csv
import os
import random
from random import shuffle
import pickle

data_path = 'DataProcessingForNB'
label_file = os.path.join(data_path, 'integer_labels.csv')


def split_train_test():
    split_ratio = 0.75
    with open(label_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')

        single_label_dict = {}
        selected = {}
        for row in spamreader:
            if len(row[2]) == 0:
                key = row[1]
                if key not in single_label_dict:
                    single_label_dict[key] = []
                else:
                    single_label_dict[key].append(row[0])
            else:
                selected[row[0]] = [int(row[1]), int(row[2])]
        tmp_keys = list(selected.keys())
        shuffle(tmp_keys)
        selected_train_key = tmp_keys[:int(len(tmp_keys) * split_ratio)]
        selected_test_key = tmp_keys[int(len(tmp_keys) * split_ratio):]
        selected_train = {}
        selected_test = {}
        for key in selected_train_key:
            selected_train[key] = selected[key]
        for key in selected_test_key:
            selected_test[key] = selected[key]

        # for item in single_label_dict:
        #
        #     shuffle(single_label_dict[item])
        #     random_train = single_label_dict[item][:int(3000*split_ratio)]
        #     random_test = single_label_dict[item][int(3000*split_ratio):3000]
        #     for tmp in random_train:
        #         selected_train[tmp] = [int(item)]
        #     for tmp in random_test:
        #         selected_test[tmp] = [int(item)]
    return selected_train, selected_test


def process2(selected, str_):
    bow_file = os.path.join(data_path, 'Presence_Vector_4k_train.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        bow_presence_dict = {}
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in selected:
                bow_presence_dict[key] = vector
    with open('bow_presence_dict_'+str_, 'bw') as f:
        pickle.dump(bow_presence_dict, f)

    bow_file = os.path.join(data_path, 'Count_Vector.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        bow_count_dict = {}
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in selected:
                bow_count_dict[key] = vector
    with open('bow_count_dict_'+str_, 'bw') as f:
        pickle.dump(bow_count_dict, f)

    bow_file = os.path.join(data_path, 'coh_mixed.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        co_matrix_dict = {}
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = []
            for t in tmp:
                if t == 'None':
                    t = 0
                else:
                    t = float(t)
                vector.append(t)
            if key in selected:
                co_matrix_dict[key] = vector
    with open('co_matrix_dict_'+str_, 'bw') as f:
        pickle.dump(co_matrix_dict, f)

    bow_file = os.path.join(data_path, 'bipolar_labels.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        bipolar_labels = {}
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in selected:
                bipolar_labels[key] = vector
    with open('bipolar_labels_'+str_, 'bw') as f:
        pickle.dump(bipolar_labels, f)



def process3(selected, str_):
    bow_file = os.path.join(data_path, 'Presence_Vector_4k_test.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        bow_presence_dict = {}
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in selected:
                bow_presence_dict[key] = vector
    with open('bow_presence_dict_'+str_, 'bw') as f:
        pickle.dump(bow_presence_dict, f)

    bow_file = os.path.join(data_path, 'Count_Vector.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        bow_count_dict = {}
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in selected:
                bow_count_dict[key] = vector
    with open('bow_count_dict_'+str_, 'bw') as f:
        pickle.dump(bow_count_dict, f)

    bow_file = os.path.join(data_path, 'coh_mixed.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        co_matrix_dict = {}
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = []
            for t in tmp:
                if t == 'None':
                    t = 0
                else:
                    t = float(t)
                vector.append(t)
            if key in selected:
                co_matrix_dict[key] = vector
    with open('co_matrix_dict_'+str_, 'bw') as f:
        pickle.dump(co_matrix_dict, f)

    bow_file = os.path.join(data_path, 'bipolar_labels.csv')
    with open(bow_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        bipolar_labels = {}
        for row in spamreader:
            key = row[0]
            tmp = row[1: -1]
            vector = [int(x) for x in tmp]
            if key in selected:
                bipolar_labels[key] = vector
    with open('bipolar_labels_'+str_, 'bw') as f:
        pickle.dump(bipolar_labels, f)


def load_variable(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # two selected dictionary for training and test
    # selected_train, selected_test = split_train_test()

    with open('train_test_ids_4k.pkl', 'br') as f:
        selected_train, selected_test = pickle.load(f)
    # Create train and test dictionarys for co matrix, cbow
    process2(selected_train, 'train')
    process3(selected_test, 'test')

    # TODO: this list convert is dangerous, not causing problem for now, but never do this again
    train_seq = list(load_variable('bipolar_labels_train').keys())
    test_seq = list(load_variable('bipolar_labels_test').keys())

    def write_to_file(file, str_):
        data = load_variable(file + str_)
        with open(file + str_ + '.txt', 'w') as f:
            if str_ == 'train':
                for key in train_seq:
                    for t in data[key]:
                        f.write(str(t) + ' ')
                    f.write('\n')
            else:
                for key in test_seq:
                    for t in data[key]:
                        f.write(str(t) + ' ')
                    f.write('\n')

    write_to_file('co_matrix_dict_', 'train')
    write_to_file('co_matrix_dict_', 'test')
    write_to_file('bipolar_labels_', 'train')
    write_to_file('bipolar_labels_', 'test')
    write_to_file('bow_count_dict_', 'train')
    write_to_file('bow_count_dict_', 'test')
    write_to_file('bow_presence_dict_', 'train')
    write_to_file('bow_presence_dict_', 'test')
    pause = 0


