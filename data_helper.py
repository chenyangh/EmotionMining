import os
import re
import sys
import json
import pickle
import itertools
import numpy as np
import pandas as pd
import gensim as gs
from collections import Counter
import logging
from gensim.models import word2vec
import gensim
from preprocess_twitter import twitter_tokenize
# logging.getLogger().setLevel(logging.INFO)
import re
import nltk
import csv

def sentence_to_word_list(a_review):
    # Use regular expressions to do a find-and-replace

    tmp = a_review.split()
    words = []
    for word in tmp:
        words.extend(clean_str(word).split())
    return words


def train_word2vec():

    def split_sentences(review, __remove_stopwords=False):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            tmp = sentence_to_word_list(raw_sentence)
            if len(tmp) > 0:
                sentences.append(tmp)
        return sentences

    def get_sentences():
        file_list = ['emotion.neg.0.txt', 'emotion.pos.0.txt']
        sentences = []
        for file in file_list:
            with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
                tmp = rf.readlines()
                for line in tmp:
                    sentences.extend(split_sentences(line))
        return sentences

    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    sentences = get_sentences()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 1   # Minimum word count
    num_workers = 9       # Number of threads to run in parallel
    context = 5          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model skip_gram...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                size=num_features, min_count=min_word_count,
                window=context, sample=downsampling, sg=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "feature/imdb_skip_gram_word2vec"
    model.save(model_name)
    print("Training model cbow...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling, sg=0)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "feature/imdb_cbow_word2vec"
    model.save(model_name)




def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9]", " ", s)
    s = re.sub(r" : ", ":", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()


def remove_punctuation(s):
    s = re.sub(r"[^A-Za-z0-9]", " ", s)
    return s.strip().lower()

#model = gensim.models.KeyedVectors.load_word2vec_format('feature/GoogleNews-vectors-negative300.bin', binary=True)

def load_pretrain_word2vec(vocabulary, embedding_option, emb_dim):
    if embedding_option == "word2vec":
        print('Loading word2vec word embedding')
        with open('feature/word2vecModel', 'br') as f:
            model = pickle.load(f)
        embed_dict = model.vocab
    elif embedding_option == "glove":
        print('Loading glove word embedding')
        with open('feature/gloveModel', 'br') as f:
            model = pickle.load(f)
        embed_dict = model.vocab
    elif embedding_option == "glovetwitter":
        print('Loading glovetwitter word embedding')
        with open('feature/glove_twitter.27B.200d', 'br') as f:
            model = pickle.load(f)
        embed_dict = model.vocab
    elif embedding_option == "fasttext":
        print('Loading fasttext word embedding')
        with open('feature/fasttextModel', 'br') as f:
            model = pickle.load(f)
        embed_dict = model.vocab

    word_embeddings = {}
    num_oov = 0
    for word in vocabulary:
        if word in embed_dict:
            vec = model.syn0[embed_dict[word].index]
            word_embeddings[word] = (vec - min(vec)) / np.add(max(vec), -min(vec)) * 2 - 1

        else:
            num_oov += 1
            word_embeddings[word] = np.random.uniform(-1, 1, emb_dim)
    print('Numb of oov is', num_oov)
    return word_embeddings


def load_random_word2vec(vocabulary):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-1, 1, 300)
    return word_embeddings



def load_embeddings(vocabulary, embedding_option='word2vec', emb_dim=300):
    # Sentiment embedding
    # emb_dict ={}
    # with open('feature/senti_emb_20.txt', 'r') as f:
    #     for line in f.readlines():
    #         pass

    # word2vec embedding
    # if embedding == 'cbow':
    word_embeddings = load_pretrain_word2vec(vocabulary, embedding_option, emb_dim)

    # word2vec model

    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None, given_pad_len=None):
    """Pad setences during training or prediction"""
    if forced_sequence_length is None:  # Train
        sequence_length = max(len(x) for x in sentences)
    else:  # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if given_pad_len is not None:
            sequence_length = given_pad_len
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:  # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences, sequence_length


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# file_list = ['emotion.neg.0.txt', 'emotion.pos.0.txt']
def load_twitter_data():
    with open('others/train_test_ids_31k.pkl', 'br') as f:
        train_ids, test_ids = pickle.load(f)
    num_classes = 9
    data_path = 'data'
    #file_list = ['CBET-double.csv', 'CBET-single.csv']
    file_list = ['mixed.csv']
    __data = []
    __label = []
    for file in file_list:
        bow_file = os.path.join(data_path, file)
        with open(bow_file, newline='', encoding='ISO-8859-1') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t')

            for row in spamreader:
                id = row[0]
                if id in train_ids or id in test_ids:
                    sent = row[1]
                    sent = clean_str(sent).split()
                    tmp = row[2:]
                    tmp = [i for i in tmp if i != '']
                    vector = [int(x) for x in tmp]
                    __data.append(sent)
                    tmp = [1 & (i in vector) for i in range(num_classes)]
                    __label.append(tmp)
    return __data, __label


def load_data():
    x_raw, y_raw = load_twitter_data()
    print(len(x_raw))
    x_raw, sequence_length = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(x_raw)

    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw)


    labels = ['negative', 'positive']
    return x, y, vocabulary, vocabulary_inv, labels


if __name__ == "__main__":
    load_data()

