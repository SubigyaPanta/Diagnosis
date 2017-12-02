from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from panda_csv_dataset_reader import PandaCsvDataserReader

# Datasets
ALL_DATA = 'breast-cancer-wisconsin.data'
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/data/input/'

def main():
    # features = tf.
    filename = DATA_FOLDER + ALL_DATA
    headers = ['code_number', 'clump_thickness', 'cell_size_uni', 'cell_shape',
               'marginal_adhesion', 'single_cell_size', 'bare_nuclei',
               'bland_chromatin', 'normal_nucleoli', 'mitoses', 'diagnosis'
               ]
    reader = PandaCsvDataserReader(filename, headers)
    train, validation, test = reader.read()
    train_features, train_label = reader.get_feature_and_label(train, 'diagnosis')
    validation_features, validation_label = reader.get_feature_and_label(validation, 'diagnosis')
    test_features, test_label = reader.get_feature_and_label(test, 'diagnosis')

    print('validation_label: ', validation_label)



# if the python interpreter is running the source file as the main program, it sets
# the special __name__ variable to have a value "__main__". If this file is being
# imported from another module, __name__ will be set to the module's name.
if __name__ == '__main__':
    main()