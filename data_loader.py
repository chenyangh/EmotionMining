

def load_csv_data(data_file):
    """"
    Load csv file given by 'data_file'
    :param data_file: 
    :return: sentences and labels, in same sequence
    """
    import csv
    label_len = 9
    sentence = []
    label = []
    with open(data_file, newline='', encoding='ISO-8859-1') as csvfile:
        rows = csv.reader(csvfile, delimiter='\t')
        for row in rows:
            sentence.append(row[0])
            tmp = [0] * label_len
            tmp3 = [row[2], row[4]]
            tmp2 = [int(x) for x in tmp3] # TODO the way of reading file is dangerous
            for t in tmp2:
                tmp[t] = 1
            label.append(tmp)
    return sentence, label


if __name__ == '__main__':
    data, label = load_csv_data('data/CBET-double.csv')
    print(len(data))
