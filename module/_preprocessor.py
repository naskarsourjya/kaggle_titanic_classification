# imports
from unittest.mock import inplace

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class OnePreprocessor:

    def __init__(self, data, x_label, y_label):

        # init
        self.train_data_x = None
        self.test_data_x = None
        self.train_data_y = None
        self.test_data_y = None

        # storage
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.metadata = {'x_label': x_label, 
                         'y_label': y_label, 
                         'sample_size': data.shape[0]}


    def split_data(self, test_size=0.2, random_state=None):
        self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y = train_test_split(self.data[self.x_label], self.data[self.y_label], test_size=test_size, random_state=random_state)

        # store metadata
        self.metadata.update({
            "random_state": random_state,
            "test_ratio": test_size*self.data.shape[0],
            "train_ratio": (1-test_size)*self.data.shape[0]
        })

        return None

    
    def data_imputer(self, num_strategy='median', cat_strategy='most_frequent'):

        # Automatically detect numeric and categorical columns
        num_cols = self.train_data_x.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = self.train_data_x.select_dtypes(include=['object', 'category']).columns


        # Numeric imputer (median is usually better for skewed data)
        if len(num_cols)>0:
            num_imputer = SimpleImputer(strategy=num_strategy)
            self.train_data_x[num_cols] = num_imputer.fit_transform(self.train_data_x[num_cols])
            self.test_data_x[num_cols] = num_imputer.transform(self.test_data_x[num_cols])


        # Categorical imputer (most frequent)
        if len(cat_cols)>0:
            cat_imputer = SimpleImputer(strategy=cat_strategy)
            self.train_data_x[cat_cols] = cat_imputer.fit_transform(self.train_data_x[cat_cols])
            self.test_data_x[cat_cols] = cat_imputer.transform(self.test_data_x[cat_cols])

        # store metadata
        self.metadata.update({
            "num_cols": list(num_cols),
            "cat_cols": list(cat_cols),
            'imputer': True,
            "num_strategy": num_strategy,
            "cat_strategy": cat_strategy
        })

        return None


    def hot_encode_categories(self):

        # Automatically detect numeric and categorical columns
        cat_cols = list(self.train_data_x.select_dtypes(include=['object', 'category']).columns)

        # Preprocessor init
        encoder = OneHotEncoder(handle_unknown="ignore")  # handle_unknown avoids errors for new categories

        # convert the categories to numpy arrays
        train_x = encoder.fit_transform(self.train_data_x[cat_cols]).toarray()
        test_x = encoder.transform(self.test_data_x[cat_cols]).toarray()

        # generate new column names
        new_cat_cols = ["hot_encoded_"+str(i) for i in range(train_x.shape[1])]

        # store new encoded columns
        self.train_data_x[new_cat_cols] = train_x
        self.test_data_x[new_cat_cols] = test_x

        # remove old categorical columns
        self.train_data_x.drop(columns=cat_cols, inplace=True)
        self.test_data_x.drop(columns=cat_cols, inplace=True)

        # save meta data
        self.metadata.update({
            'one_hot_encoded': True
        })

        return None
    

    def preprocess_all(self, pipeline, desc_dict={}):

        # transform data
        train_x = pipeline.fit_transform(self.train_data_x)
        test_x = pipeline.transform(self.test_data_x)

        # generate new column names
        new_cat_cols = ["preprocessed_" + str(i) for i in range(train_x.shape[1])]

        # convert to dataframe and store
        self.train_data_x = pd.DataFrame(train_x, columns=new_cat_cols)
        self.test_data_x = pd.DataFrame(test_x, columns=new_cat_cols)

        # save meta data
        self.metadata.update({
            'custom_pipeline': True
        })
        self.metadata.update(desc_dict)

        return None
