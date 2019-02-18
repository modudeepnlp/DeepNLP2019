import json
import numpy as np
import tensorflow as tf
import tarfile
import tempfile
import json
import os
import re
import sys

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  print(max(len(x.split()) for x in left))
  print(max(len(x.split()) for x in right))

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = to_categorical(Y, len(LABELS))

  return left, right, Y

# data_path = 'data_in/snli/'

# training = get_data(data_path+'snli_1.0_train.jsonl')
# validation = get_data(data_path+'snli_1.0_dev.jsonl')
# test = get_data(data_path+'snli_1.0_test.jsonl')

# tokenizer = Tokenizer(lower=False, filters='')
# tokenizer.fit_on_texts(training[0] + training[1])

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
# VOCAB = len(tokenizer.word_counts) + 1
# LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

# MAX_LEN = 42

# to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
# prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

# training = prepare_data(training)
# validation = prepare_data(validation)
# test = prepare_data(test)

# print(training)


# print('Build model...')
# print('Vocab size =', VOCAB)

# GLOVE_STORE = 'precomputed_glove.weights'
# if USE_GLOVE:
#   if not os.path.exists(GLOVE_STORE + '.npy'):
#     print('Computing GloVe')
  
#     embeddings_index = {}
#     f = open(data_path+'glove.840B.300d.txt')
#     for line in f:
#       values = line.split(' ')
#       word = values[0]
#       coefs = np.asarray(values[1:], dtype='float32')
#       embeddings_index[word] = coefs
#     f.close()
    
#     # prepare embedding matrix
#     embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
#     for word, i in tokenizer.word_index.items():
#       embedding_vector = embeddings_index.get(word)
#       if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#       else:
#         print('Missing from GloVe: {}'.format(word))
  
#     np.save(GLOVE_STORE, embedding_matrix)

#   print('Loading GloVe')
#   embedding_matrix = np.load(GLOVE_STORE + '.npy')

#   print('Total number of null word embeddings:')
#   print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

#   embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
# else:
#   embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)