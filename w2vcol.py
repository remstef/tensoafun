#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:23:11 2017

@author: rem

based on https://github.com/tensorflow/tensorflow/blob/7d5c426692ba03ebfd67afc03c967c594f036932/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
and https://raw.githubusercontent.com/adventuresinML/adventures-in-ml-code/8c13d62c7a73ab7463718fb1f78274fcc969628e/tf_word2vec.py


"""

import sklearn.preprocessing
import math
import datetime as dt
import sys
import gzip

import numpy as np
import tensorflow as tf

#%% test variables
testfile = '/Users/rem/data/wiki.en.simple/simplewikipedia_fruits_1skip3gram.txt'

#%% readfile 
def readfile(fin):
    i = 0;
    word = []
    feat = []
    with sys.stdin if not fin else gzip.open(fin) if fin.endswith('.gz') else open(fin) as f:
        for line in f:
            i += 1
            if i % 10000 == 0:
                print('read {} lines'.format(i), file=sys.stderr)
            line = line.strip()
            w, context = line.split('\t')
            for f in context.split(' '):
                word.append(w)
                feat.append(f)
    print('read {} lines'.format(i), file=sys.stderr);
    print('extracted {} ({}) word feature pairs'.format(len(word), len(feat)), file=sys.stderr);
    return (word, feat)

#%% prepare data
def prepare_data(word_feature_file):
    
    word_encoder = sklearn.preprocessing.LabelEncoder()
    context_encoder = sklearn.preprocessing.LabelEncoder()
    
    words, contexts = readfile(word_feature_file)

    word_data = word_encoder.fit_transform(words)    
    context_data = context_encoder.fit_transform(contexts)

    # free some space    
    del words, contexts

    word_dim = len(word_encoder.classes_)
    context_dim = len(context_encoder.classes_)
    
    return word_dim, context_dim, word_data, context_data, word_encoder, context_encoder

#%% generate batch data
def generate_batch_data(word_data, context_data, batch_size, offset):
    
    # generate batch indices, start at zero if end of data was reached
    offset_indices = [i % len(word_data) for i in range(offset,offset+batch_size)]
    input_batch_data = word_data[offset_indices]
    context_batch_data = np.zeros(shape=(batch_size, 1), dtype=np.int32)
    context_batch_data[:,0] = context_data[offset_indices]
    
    # new offset    
    offset = (offset + batch_size) % len(word_data)
    return input_batch_data, context_batch_data, offset


#%%
def build_graph(vocabulary_size, context_size, embedding_size, batch_size, validation_size, learning_rate, num_nce_samples = -1):
    
    # Input data
    train_input_batch = tf.placeholder(tf.int32, shape=[batch_size])
    train_context_batch = tf.placeholder(tf.int32, shape=[batch_size, 1])
    validation_input = tf.placeholder(tf.int32, shape=[validation_size])

    # Look up embeddings for inputs.
    embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
    # validation: 
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, validation_input)
    similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)    
    
    train_embed_batch = tf.nn.embedding_lookup(embeddings, train_input_batch)
    
    if num_nce_samples < 1: # use sgd
        # Construct the variables for the softmax
        weights = tf.Variable(
                tf.truncated_normal([embedding_size, context_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        biases = tf.Variable(tf.zeros([context_size]))
        
        # hidden layer function
        hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(train_embed_batch))) + biases

        # convert train_context to a one-hot format
        train_output_batch_one_hot = tf.one_hot(train_context_batch, context_size)

        # optimize cross entropy
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_output_batch_one_hot))

        # Construct the SGD optimizer using a learning rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    
        return train_input_batch, train_context_batch, validation_input, cross_entropy, optimizer, similarity, normalized_embeddings
    
    else: # use nce
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
                tf.truncated_normal([context_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([context_size]))

        nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(
                        weights     = nce_weights,
                        biases      = nce_biases,
                        labels      = train_context_batch,
                        inputs      = train_embed_batch,
                        num_sampled = num_nce_samples,
                        num_classes = context_size))
    
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(nce_loss)
        
    return train_input_batch, train_context_batch, validation_input, nce_loss, optimizer, similarity, normalized_embeddings
    
#%%
def main():

    batch_size = 128
    embedding_size = 300  # Dimension of the embedding vector.
    learning_rate = 1.0

    validate = ['apple','tree','leaf', 'pineapple'] # this is not a real validation, more like a visual suuport to see progress 
    num_nce_samples = 0    # Number of negative examples to sample.
    iterations = 50000 # specify how many steps for optimization
    min_epochs = 5 # specify the minimum number of passes over the whole dataset
    
    word_dim, \
    context_dim, \
    word_data, \
    context_data, \
    word_encoder, \
    context_encoder = prepare_data(testfile)

    validation_data = word_encoder.transform(validate)
    
    with tf.device('/cpu'):
        
        train_input_batch, \
        train_context_batch, \
        validation_input, \
        loss, \
        optimizer, similarity, \
        normalized_embeddings = build_graph(
                word_dim, 
                context_dim,
                embedding_size, 
                batch_size, 
                len(validate), 
                learning_rate, 
                num_nce_samples)
    
        with tf.Session() as sess:
            
            # We must initialize all variables before we use them.
            sess.run(tf.global_variables_initializer())
            print('Initialized')
    
            average_loss = 0
            batch_offset = 0
            num_epochs = 0
            for iteration in range(iterations):
                if batch_offset < batch_size:
                    num_epochs += 1
                    print('epoch %d' % num_epochs)
              
                input_batch, context_batch, batch_offset = generate_batch_data(
                        word_data, context_data, batch_size, batch_offset)
              
                # feed the graph with the batch data
                feed_dict = { 
                        train_input_batch: input_batch, 
                        train_context_batch: context_batch}
    
                # We perform one update step by evaluating the optimizer op, including it
                # in the list of returned values for session.run()
                _, loss_value = sess.run(
                        fetches = [optimizer, loss], 
                        feed_dict = feed_dict)
                
                average_loss += loss_value
                if iteration % 2000 == 0:
                    average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at iteration ', iteration, ': ', average_loss)
                    average_loss = 0
    
                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if iteration % 10000 == 0:
                    sim = sess.run(fetches = similarity, feed_dict = {validation_input: validation_data})
                    for i in range(len(validate)):
                        valid_word = validate[i]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = word_encoder.inverse_transform([nearest[k]])
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)
                            
            final_embeddings = normalized_embeddings.eval()
       
#%% run the script
if __name__ == "__main__":
    main()     
        

#num_steps = 100
#softmax_start_time = dt.datetime.now()
#run(graph, num_steps=num_steps)
#softmax_end_time = dt.datetime.now()
#print("Softmax method took {} minutes to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))
#
#with graph.as_default():
#
#    # Construct the variables for the NCE loss
#    nce_weights = tf.Variable(
#        tf.truncated_normal([vocabulary_size, embedding_size],
#                            stddev=1.0 / math.sqrt(embedding_size)))
#    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
#
#    nce_loss = tf.reduce_mean(
#        tf.nn.nce_loss(weights=nce_weights,
#                       biases=nce_biases,
#                       labels=train_context,
#                       inputs=embed,
#                       num_sampled=num_sampled,
#                       num_classes=vocabulary_size))
#
#    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)
#
#    # Add variable initializer.
#    init = tf.global_variables_initializer()
#
#num_steps = 50000
#nce_start_time = dt.datetime.now()
#run(graph, num_steps)
#nce_end_time = dt.datetime.now()
#print("NCE method took {} minutes to run 100 iterations".format((nce_end_time-nce_start_time).total_seconds()))
