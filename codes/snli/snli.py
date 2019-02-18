# coding: utf-8

import sys
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import pickle
import glob

from sklearn.model_selection import train_test_split
from dataset.readers.snli_import import get_data

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

import json

batch_size = 512
MAX_LEN = 42
EPOCHS = 5

os.environ["CUDA_VISIBLE_DEVICES"]="0" #For TEST
tf.logging.set_verbosity("INFO")

data_path = 'data_in/snli/'

training = get_data(data_path+'snli_1.0_train.jsonl')
validation = get_data(data_path+'snli_1.0_dev.jsonl')
test = get_data(data_path+'snli_1.0_test.jsonl')

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

## 미리 Global 변수를 지정하자. 파일 명, 파일 위치, 디렉토리 등이 있다.

DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'

## 학습에 필요한 파라메터들에 대해서 지정하는 부분이다.


RNN = None

BATCH_SIZE = 512
HIDDEN = 128
BUFFER_SIZE = 1000000

NUM_LAYERS = 3
DROPOUT_RATIO = 0.3

TEST_SPLIT = 0.1
RNG_SEED = 13371447
EMBEDDING_DIM = 300
MAX_SEQ_LEN = 42

WORD_EMBEDDING_DIM = 100
CONV_FEATURE_DIM = 300
CONV_OUTPUT_DIM = 128
CONV_WINDOW_SIZE = 3
DROPOUT_RATIO = 0.5
SIMILARITY_DENSE_FEATURE_DIM = 200

LAYERS = 1
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 512
PATIENCE = 4 # 8
MAX_EPOCHS = 42
MAX_LEN = 42
DP = 0.2
L2 = 4e-6
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'

def data_len(df_list):
    q_data_len = np.array([min(len(x), MAX_SEQ_LEN) for x in df_list], dtype=np.int32)
    return q_data_len

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)

prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])
len_data = lambda data: (data_len(data[0]), data_len(data[1]))

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

training_len = len_data(training)
validation_len = len_data(validation)
test_len = len_data(test)

def save_pickle(file_nm, var_nm):
    with open(file_nm, 'wb') as fp:
        pickle.dump(var_nm, fp)
        print("save {}".format(file_nm))

def load_pickle(file_nm):
    with open(file_nm, 'rb') as fp:
        pkl = pickle.load(fp)
        print("load {}".format(file_nm))
        return pkl 

# save_pickle(data_path + 'training.pkl', training)
# save_pickle(data_path + 'validation.pkl',validation)
# save_pickle(data_path + 'test.pkl', test)

# save_pickle(data_path + 'training_len.pkl', training_len)
# save_pickle(data_path + 'validation_len.pkl',validation_len)
# save_pickle(data_path + 'test_len.pkl', test_len)

# training = load_pickle(data_path + 'training.pkl')
# validation = load_pickle(data_path + 'validation.pkl')
# test = load_pickle(data_path + 'test.pkl')

# training_len = load_pickle(data_path + 'training_len.pkl')
# validation_len = load_pickle(data_path + 'validation_len.pkl')
# test_len = load_pickle(data_path + 'test_len.pkl')

USE_GLOVE = True
TRAIN_EMBED = False

def train_input_fn():
    train_dataset = tf.data.Dataset.from_tensor_slices((training[0], training[1], training_len[0],
                                                        training_len[1], training[2])).shuffle(
                buffer_size=BUFFER_SIZE).prefetch(buffer_size=batch_size).batch(batch_size).repeat(EPOCHS)
    iterator = train_dataset.make_one_shot_iterator()
    q1, q2, q1_len, q2_len, labels = iterator.get_next()
    features = {'q1': q1, "q2": q2, "q1_len": q1_len, "q2_len": q2_len}
    # labels = {'labels': labels}
    return features, labels

def valid_input_fn():
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation[0], validation[1], validation_len[0], 
                                                        validation_len[1], validation[2])).shuffle(
                    buffer_size=BUFFER_SIZE).prefetch(buffer_size=batch_size).batch(batch_size).repeat(EPOCHS)
    iterator = validation_dataset.make_one_shot_iterator()
    q1, q2, q1_len, q2_len, labels = iterator.get_next()
    features = {'q1': q1, "q2": q2, "q1_len": q1_len, "q2_len": q2_len}
    # labels = {'labels': labels}

    return features, labels

GLOVE_STORE = 'precomputed_glove.weights'
if USE_GLOVE:
    if not os.path.exists(GLOVE_STORE + '.npy'):
        print('Computing GloVe')
    
        embeddings_index = {}
        f = open(data_path+'glove.840B.300d.txt')
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        
        # prepare embedding matrix
        embedding_matrix = np.zeros((VOCAB, EMBEDDING_DIM))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                print('Missing from GloVe: {}'.format(word))
    
        np.save(GLOVE_STORE, embedding_matrix)

    print('Loading GloVe')
    embedding_matrix = np.load(GLOVE_STORE + '.npy')

    print('Total number of null word embeddings:')
    print(np.sum(np.sum(embedding_matrix, axis=1) == 0))


def basic_conv_sementic_network(inputs, name):
    conv_layer = tf.keras.layers.Conv1D(CONV_FEATURE_DIM, 
                                        CONV_WINDOW_SIZE, 
                                        activation=tf.nn.relu, 
                                        name=name + 'conv_1d',
                                        padding='same')(inputs)
    
    max_pool_layer = tf.keras.layers.MaxPool1D(MAX_SEQ_LEN, 
                                               1)(conv_layer)

    output_layer = tf.keras.layers.Dense(CONV_OUTPUT_DIM, 
                                         activation=tf.nn.relu,
                                         name=name + 'dense')(max_pool_layer)
    output_layer = tf.squeeze(output_layer, 1)
    
    return output_layer

def estimator_model(features, labels, mode):

        
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    
    if USE_GLOVE:
        embed = Embedding(VOCAB, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
    else:        
        embed = Embedding(VOCAB, EMBEDDING_DIM, input_length=MAX_LEN)

    prem = embed(features['q1'])
    hypo = embed(features['q2'])

    rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)
    SumEmbeddings = layers.Lambda(lambda x: keras.backend.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))    
    
    translate = layers.TimeDistributed(layers.Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    prem = translate(prem)
    hypo = translate(hypo)

    if RNN and LAYERS > 1:
        for l in range(LAYERS - 1):
            rnn = RNN(return_sequences=True, **rnn_kwargs)
            prem = layers.BatchNormalization()(rnn(prem))
            hypo = layers.BatchNormalization()(rnn(hypo))
    
    rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
    prem = rnn(prem)
    hypo = rnn(hypo)
    prem = layers.BatchNormalization()(prem)
    hypo = layers.BatchNormalization()(hypo)

    joint = keras.layers.concatenate([prem, hypo])
    joint = layers.Dropout(DP)(joint)
    for i in range(3):
        joint = layers.Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION)(joint)
        joint = layers.Dropout(DP)(joint)
        joint = layers.BatchNormalization()(joint)    
    
    	
    # """ For Conv """
    # base_sementic_matrix = basic_conv_sementic_network(base_embedded_matrix, 'base')
    # hypothesis_sementic_matrix = basic_conv_sementic_network(hypothesis_embedded_matrix, 'hypothesis')

    # base_sementic_matrix = tf.keras.layers.Dropout(DROPOUT_RATIO)(query)
    # hypothesis_sementic_matrix = tf.keras.layers.Dropout(DROPOUT_RATIO)(sim)    
    
    # merged_matrix = tf.concat([base_sementic_matrix, hypothesis_sementic_matrix], -1)

    # similarity_dense_layer = tf.keras.layers.Dense(250,
    #                                          activation=tf.nn.relu)(merged_matrix)
    
    # similarity_dense_layer = tf.keras.layers.Dropout(DROPOUT_RATIO)(similarity_dense_layer)
    
    with tf.variable_scope('output_layer'):
        # pred = tf.keras.layers.Dense(len(LABELS), activation='softmax')(similarity_dense_layer)
        pred = tf.keras.layers.Dense(len(LABELS), activation='softmax')(join)
        print("prediction: {}".format(pred))
    
    if PREDICT:
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  predictions={
                      'is_duplicate': pred
                  })
    
    #prediction 진행 시, None
    if labels is not None:
        labels = tf.to_float(labels)
    
    def loss_fn(logits, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=labels))
        # loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        return loss
    
    def evaluate(logits, labels):
        # accuracy = tf.metrics.accuracy(labels, tf.round(logits))

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return acc

    loss = loss_fn(pred, labels)
    acc = evaluate(pred, labels)

    logging_hook = tf.train.LoggingTensorHook({"loss" : loss,  "accuracy" : acc}, every_n_iter=100)
    
    if EVAL:
        # acc = evaluate(pred, labels)
        accuracy = tf.metrics.accuracy(labels, tf.round(pred))
        eval_metric_ops = {'acc': accuracy}
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  eval_metric_ops= eval_metric_ops,
                  loss=loss)

    elif TRAIN:
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  train_op=train_op,
                  loss=loss,
                  training_hooks= [logging_hook])

model_dir = os.path.join(os.getcwd(), DATA_OUT_PATH + "/checkpoint/model/")
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig(save_checkpoints_steps=500,
                                save_checkpoints_secs=None,
                                  keep_checkpoint_max=2,
                                  log_step_count_steps=100)

model_est = tf.estimator.Estimator(estimator_model, model_dir=model_dir, config=config_tf)

model_est.train(train_input_fn)
model_est.evaluate(valid_input_fn)