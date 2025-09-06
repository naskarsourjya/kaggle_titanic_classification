import pandas as pd
from module import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv('data/train.csv')

classifier = OneClassifier()
classifier.load_data(data=df_train, x_label=['Pclass'], y_label=['Survived'])
classifier.split_data(test_size=0.2, random_state=42)
classifier.set_models(model_list=[RandomForestClassifier()])
classifier.train_models()
results = classifier.evaluate_models()