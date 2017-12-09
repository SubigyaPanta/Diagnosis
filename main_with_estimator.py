from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from panda_csv_dataset_reader import PandaCsvDataserReader
from tensorflow.python import debug as tf_debug

# This file is copy of main_with_one_hot_encoding_cross_entropy just to add summary
ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = ROOT_FOLDER + '/data/'
ALL_DATA = 'input/breast-cancer-wisconsin.data'
SUMMARY_FOLDER = ROOT_FOLDER + '/summary/'

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

    # Parameters
    training_epoch = 1500 # what is this ? => How many times should we train the model with complete training data
    label_values = {2: np.array([0]), 4: np.array([1])}

    # Features
    feature_columns = [tf.feature_column.numeric_column('x', shape=[9])]
    unique_train = [2,4]

    # Type of classifier
    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, model_dir= SUMMARY_FOLDER + 'model', n_classes=2)

    # Define training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(train_features.values)},
        y=get_label_values_from_label(train_label, label_values),
        num_epochs=training_epoch,
        shuffle=False
    )

    # Train Model
    classifier.train(input_fn=train_input_fn)

    # Define test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_features.values},
        y=get_label_values_from_label(test_label, label_values),
        num_epochs= 1,
        shuffle=False
    )

    # Evaluate accuracy
    accuracy = classifier.evaluate(input_fn=test_input_fn)['accuracy']
    print('accuracy', accuracy)


def get_label_values_from_label(label, label_values)->np:
    y_val = []
    for val in label.values:
        y_val.append(label_values[val])
    return np.array(y_val)

# if the python interpreter is running the source file as the main program, it sets
# the special __name__ variable to have a value "__main__". If this file is being
# imported from another module, __name__ will be set to the module's name.
if __name__ == '__main__':
    main()