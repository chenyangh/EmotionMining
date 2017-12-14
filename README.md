# Emotion Mining Project

To train the experiment.
```python3 train.py```
Tensorflow version 0.12 is required. The evaluation on develop set and test can be given.


Structure
```
.
+-- train.py    # file to train the deep learning model
+-- text_cnn_rnn.py # DL models are defined there
+-- data_healper.py # contains wordembedding and data pre-processing
+-- naive_bayes.py  # naive bayes model
+-- preprocess_twitter.py # pre-processing for twitter corpus used by Glove
+-- measurements.py # F-macro, F-micro, Exam-F are given
+-- feature
|   +-- fasttexModel    # Facebook fastText 
|   +-- gloveModel      # glove on Wikipedia
|   +-- word2vecModel   # word2vec on GoogNews
|   +-- glove_twitter.27B.200d # glove on twitter
+-- data
|   +-- mixed.csv
|   +-- integer_labels.csv
|   +-- bipolar_labels.csv
|   +-- binary_labels.csv
|   +-- Presence_Vector_31k_test.csv
|   +-- Presence_vector_31k_train.csv
|-- others
|   +-- train_test_ids_4k.pkl
|   +-- train_test_ids_31k.pkl
```


## Word Embedding supported

    * GloVe on Wikipedia
    * GloVe on twitter
    * word2vec
    * fastText

## Deep Learning Model Supported
TextCNNRNN, TextRNN, TextCNN, TextBiRNN, TextCNNBiRNN
   * CNN-RNN(GRU/LSTM)
   * CNN
   * RNN (GRU/LSTM)
   * CNN-biRNN(GRU/LSTM)
   
## Settings
`training_confg.json` provides some part of the parameters tuning for a model. But switching model, change pre-processing 
script, such adjustment have to be done at code level.
An example of setting is,
```
{
    "batch_size": 64,
    "dropout_keep_prob": 0.5,
    "embedding_dim": 300,
    "evaluate_every": 200,
    "filter_sizes": "3,4,5",
    "hidden_unit": 300,
    "l2_reg_lambda": 0.0,
    "max_pool_size": 2,
    "non_static": false,
    "num_epochs": 5,
    "num_filters": 256,
    "embedding": "fasttext"
}
```
option of embedding including 'word2vec', 'glove', 'fasttext' and 'glovetwitter'. Note that except for 'glovetwitter', 
or other embeddings are 300 dimensional. Need to change 'embedding_dim' to 200 before using 'glovetwitter'.


