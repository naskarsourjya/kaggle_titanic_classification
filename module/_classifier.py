# imports
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from catboost import CatBoostClassifier

class OneClassifier:
    def __init__(self, file_name, OnePreprocessor):
        self.model_list = None
        self.OnePreprocessor = OnePreprocessor
        self.X_preprocessed = None
        self.y_preprocessed = None
        self.file_name = file_name
        self.models_dict = None

    
    def set_models(self, models_dict):

        # assert that all keys are lists with the same length
        assert all(isinstance(v, list) for v in models_dict.values()), "All values in models_dict must be lists!"
        assert len(set(len(v) for v in models_dict.values())) == 1, "All lists in models_dict must have the same length!"
        assert 'models' in models_dict, "Key 'models' must be present in models_dict!"

        
        self.models_dict = models_dict


    def train_models(self):
        X_train = self.OnePreprocessor.train_data_x
        y_train = self.OnePreprocessor.train_data_y
        for model in self.models_dict['models']:
            if isinstance(model, CatBoostClassifier):
                cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
                model.fit(X_train, y_train, cat_features=list(cat_cols))
            else:
                model.fit(X_train, y_train)
    
    def evaluate_models(self, save_results = True):

        # init
        result_dict = {'datetime': [], 'model': [], 'TP': [], 'TN': [], 'FP': [], 'FN': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

        X_test = self.OnePreprocessor.test_data_x
        y_test = self.OnePreprocessor.test_data_y

        for model in self.models_dict['models']:
            y_pred = model.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # append results to dict
            result_dict['datetime'].append(datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
            result_dict['model'] = model.__class__.__name__
            result_dict['TP'].append(tp)
            result_dict['TN'].append(tn)
            result_dict['FP'].append(fp)
            result_dict['FN'].append(fn)
            result_dict['accuracy'].append(accuracy)
            result_dict['f1'].append(f1)
            result_dict['precision'].append(precision)
            result_dict['recall'].append(recall)


        # add preprocessing metadata
        for key in self.OnePreprocessor.metadata.keys():
            value = self.OnePreprocessor.metadata[key]
            result_dict[key] = [value] * len(self.models_dict['models'])
        
        # convert dict to dataframe
        output_dict = self.models_dict | result_dict
        result_df = pd.DataFrame(output_dict)

        # save results
        if save_results:
            self.save_results(result_dict=output_dict)

        return result_df
    

    def save_results(self, result_dict):

        # init
        file_name = self.file_name

        # open file if already exists
        try:
            df = pd.read_csv(file_name)
        except FileNotFoundError:
            df = pd.DataFrame()

        # check if df is empty, assign a new column name 'Serial No.' and add it to all dicts
        if df.empty:
            result_dict['Serial No.'] = [i+1 for i in range(len(result_dict['datetime']))]

        # df is not empty, assign the next serial number
        else:
            last_serial_no = df['Serial No.'].max()
            result_dict['Serial No.'] = [i + 1 + last_serial_no for i in range(len(result_dict['datetime']))]

        # Convert the list of dictionaries to a DataFrame
        new_df = pd.DataFrame(result_dict)

        # put the 'Serial No.' as the first column
        new_df = new_df[['Serial No.'] + [col for col in new_df.columns if col != 'Serial No.']]

        # Append the new DataFrame to the existing one
        df = pd.concat([df, new_df], ignore_index=True)

        # Save the updated DataFrame to the CSV file
        df.to_csv(file_name, index=False)

        return None

