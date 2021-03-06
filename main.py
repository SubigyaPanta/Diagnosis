from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from panda_csv_dataset_reader import PandaCsvDataserReader
import pandas as pd
# df = pd.DataFrame({"pear": [1,2,3], "apple": [2,3,4], "orange": [3,4,5]})
# print(df)
# Datasets
ALL_DATA = 'breast-cancer-wisconsin.data'
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/data/input/'

def main():
    # Import data
    filename = DATA_FOLDER + ALL_DATA
    # not include code_number as it is not a feature. and not read first column from csv file too
    headers = ['code_number', 'clump_thickness', 'cell_size_uni', 'cell_shape',
               'marginal_adhesion', 'single_cell_size', 'bare_nuclei',
               'bland_chromatin', 'normal_nucleoli', 'mitoses', 'diagnosis'
               ]
    reader = PandaCsvDataserReader(filename, headers)
    train, validation, test = reader.read(skip_columns='code_number')
    train_features, train_label = reader.get_feature_and_label(train, 'diagnosis')
    validation_features, validation_label = reader.get_feature_and_label(validation, 'diagnosis')
    test_features, test_label = reader.get_feature_and_label(test, 'diagnosis')
    # print(train_features['clump_thickness'], train_features['cell_shape'])
    # Parameters
    learning_rate = 0.01
    batch_size = 100 # what is this ? => How many data should we use at once
    training_epoch = 500 # what is this ? => How many times should we train the model with complete training data
    display_step = 1 # what is this ?
    label_values = {2: np.array([1, 0]), 4: np.array([0, 1])}
    # print(label_values[4])
    # print(type(train_features))
    # print(train_label.as_matrix(columns=)) # to converto to numpy arrays
    # print(train_features.values)
    # a = pd.Series(train_label.values)
    # print('a: ', a)
    # print('lable.ds ', label_values[(val for val in train_label.values)])
    # print('get_lable', get_label_values_from_label(train_label, label_values))
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 9]) # there are 9 features, the first one is id which is not a feature
    y = tf.placeholder(tf.float32, [None, 1])
    print('x_shape', x.shape, train_features.shape)
    c = np.split(train_label.values, train_label.size)
    bb = np.asarray(c)
    print('y_shape', y.shape, bb.shape)
    print('bb', bb)
    # Set Model weights
    W = tf.Variable(tf.zeros([9,1])) # 9 rows and 2 columns
    b = tf.Variable(tf.zeros([1])) # equivalent to 1 row 2 columns

    # Model
    # pred = tf.nn.sigmoid(tf.matmul(x, W) + b)
    # pred = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b) # more stable
    pred = tf.matmul(x, W) + b

    # Calculate loss or cost using cross-entropy One very common, very nice
    # function to determine the loss of a model is called "cross-entropy."
    # Cross-entropy arises from thinking about information compressing codes in
    # information theory but it winds up being an important idea in lots of areas,
    # from gambling to machine learning. It's defined as:
    #       summation pred*log(y)
    # Where, pred is our predicted probability distribution, and y is the true distribution
    #  (the one-hot vector with the digit labels). In some rough sense, the cross-entropy
    #  is measuring how inefficient our predictions are for describing the truth.
    #
    # cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    cost = tf.reduce_mean(tf.squared_difference(y, pred))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

    # Perform optimization
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize the variables
    init = tf.global_variables_initializer()
    # input_x = tf.estimator.inputs.numpy_input_fn()
    # Start training
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epoch):
            # avg_cost = 0
        #     total_batch = train_features.shape[0] / 100
            _, c = sess.run([optimizer, cost], feed_dict={x: train_features.values, y: bb })
        #     _, c = sess.run([optimizer, cost], feed_dict={x: train_features.values,
        #                                                   y: get_label_values_from_label(train_label, label_values)})
            # avg_cost +=
            if (epoch+1) % display_step == 0:
                print('Epoch: ', '%04d' % (epoch + 1), 'Cost: ', '{:.9f}'.format(c), 'W=', W.eval(), 'b=', b.eval())
                print(c)

        print('Optimization Finished!')

        # Evaluate accuracy
        weight = W.eval();
        bias = b.eval();
        yy = np.matmul(test_features, weight) + bias;
        print('test:', yy, test_label);

def get_label_values_from_label(label, label_values)->np:
    y_val = []
    for val in label.values:
        y_val.append(label_values[val])
    return np.array(y_val)

# def change_label_to_numpy_array_label(label:np):
    # for val in label.
# if the python interpreter is running the source file as the main program, it sets
# the special __name__ variable to have a value "__main__". If this file is being
# imported from another module, __name__ will be set to the module's name.
if __name__ == '__main__':
    main()