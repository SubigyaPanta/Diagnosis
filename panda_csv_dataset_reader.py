
import numpy as np
import pandas as pd

class PandaCsvDataserReader:

    def __init__(self, file, headers:list, train_percent=0.6, validate_percent=0.2, seed=None):
        self.file = file
        self.headers = headers
        self.train_percent = train_percent
        self.validate_percent = validate_percent
        self.seed = seed

    def read(self, skip_columns=None):
        self.load_data()
        self.remove_columns_if_required(skip_columns)
        return self.split_train_validate_test(self.data)

    def load_data(self):
        # load into pandas dataframe
        self.data = pd.read_csv(self.file, names=self.headers, na_values='?')
        # delete rows with na values
        self.data.dropna(inplace=True)
        return self.data

    def split_train_validate_test(self, data):
        """ Returns train, validate and test data set in ration 60%, 20%, 20% of dataset"""
        np.random.seed(self.seed)
        perm = np.random.permutation(data.index)
        length = len(data)
        train_end = int(self.train_percent * length)
        validate_end = int(self.validate_percent * length) + train_end
        self.train = data.ix[perm[:train_end]]
        self.validate = data.ix[perm[train_end:validate_end]]
        self.test = data.ix[perm[validate_end:]]

        return self.train, self.validate, self.test

    def get_feature_and_label(self, dataset, label_name)-> (pd.DataFrame, pd.DataFrame):
        """ label_name is the Y value in the dataset. Eg. dataset with A, B, C
        features and D col as output label
        It will separate features and labels and return it.
        """
        dataset_labels = dataset.pop(label_name)
        return dataset, dataset_labels

    def delete_column(self, dataset: pd.DataFrame, column_name) -> pd.DataFrame:
        return dataset.drop(column_name, axis=1, inplace=True)

    def remove_columns_if_required(self, column_names:list):
        if not column_names:
            return
        else:
            self.delete_column(self.data, column_names)

