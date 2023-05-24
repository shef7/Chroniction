# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv(r'C:\Chroniction\Multi_Disease_Predictor\models\heart.csv')

# Renaming HeartDiseasePedigreeFunction as DPF
df = df.rename(columns={'HeartDiseasePedigreeFunction':'BCPF'})


df_copy = df.copy(deep=True)
df_copy[['thalach','restecg','cp','trestbps','chol']] = df_copy[['thalach','restecg','cp','trestbps','chol']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['thalach'].fillna(df_copy['thalach'].mean(), inplace=True)
df_copy['restecg'].fillna(df_copy['restecg'].mean(), inplace=True)
df_copy['cp'].fillna(df_copy['cp'].median(), inplace=True)
df_copy['trestbps'].fillna(df_copy['trestbps'].median(), inplace=True)
df_copy['chol'].fillna(df_copy['chol'].median(), inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'heart.pkl'
pickle.dump(classifier, open(filename, 'wb'))