

def load_csv_key(data_file):
    """"
    Load csv file given by 'data_file'
    :param data_file: 
    :return: will be a list of keys (first column)
    """
    import csv
    with open(data_file, newline='', encoding='ISO-8859-1') as csvfile:
        rows = csv.reader(csvfile, delimiter='\t')
        key = []
        for row in rows:
            key.append(row[0])
    return key


if __name__ == '__main__':
    data = load_csv_key('data/CBET-double.csv')
    print(len(data))