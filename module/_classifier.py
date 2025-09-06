# imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

class OneClassifier:
    def __init__(self):
        self.model_list = None
        self.data = None
        self.x_label = None
        self.y_label = None
        self.X_preprocessed = None
        self.y_preprocessed = None


    def load_data(self, data, x_label, y_label):
        self.data = data
        self.x_label = x_label
        self.y_label = y_label

    
    def set_models(self, model_list):
        self.model_list = model_list

    
    def split_data(self, test_size=0.2, random_state=None):
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)


    def train_models(self):
        X_train = self.train_data[self.x_label]
        y_train = self.train_data[self.y_label]
        for model in self.model_list:
            model.fit(X_train, y_train)
    
    def evaluate_models(self):

        # init
        result_dict = {'model': [], 'TP': [], 'TN': [], 'FP': [], 'FN': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

        X_test = self.test_data[self.x_label]
        y_test = self.test_data[self.y_label]

        for model in self.model_list:
            y_pred = model.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            result_dict['model'].append(model.__class__.__name__)
            result_dict['TP'].append(tp)
            result_dict['TN'].append(tn)
            result_dict['FP'].append(fp)
            result_dict['FN'].append(fn)
            result_dict['accuracy'].append(accuracy)
            result_dict['f1'].append(f1)
            result_dict['precision'].append(precision)
            result_dict['recall'].append(recall)
        
        # convert dict to dataframe
        result_df = pd.DataFrame(result_dict)

        return result_df

