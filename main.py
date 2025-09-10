import pandas as pd
from module import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df_train = pd.read_csv('data/train.csv')


preprocess_pipeline = Pipeline([
    ("scaler", StandardScaler()),   # Step 1: scale
    ("pca", PCA(n_components=0.95))    # Step 2: PCA
])

preprocessor = OnePreprocessor(data=df_train, x_label=['Pclass', 'Cabin'], y_label=['Survived'])
preprocessor.split_data(test_size=0.2)
preprocessor.data_imputer(num_strategy='median', cat_strategy='most_frequent')
preprocessor.hot_encode_categories()
preprocessor.preprocess_all(pipeline=preprocess_pipeline)


models_dict = {'models': [CatBoostClassifier()]}


classifier = OneClassifier(file_name='results_dummy.csv', OnePreprocessor=preprocessor)
classifier.set_models(models_dict = models_dict)
classifier.train_models()
classifier.evaluate_models()