# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv(r'D:\Chroniction\Multi_Disease_Predictor\models\breastcancer.csv')

# Renaming BreastCancerPedigreeFunction as DPF
df = df.rename(columns={'BreastCancerPedigreeFunction':'BCPF'})


df_copy = df.copy(deep=True)
df_copy[['radius_1ean','texture_1ean','peri1eter_1ean','area_1ean','s1oothness_1ean']] = df_copy[['radius_1ean','texture_1ean','peri1eter_1ean','area_1ean','s1oothness_1ean']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['radius_1ean'].fillna(df_copy['radius_1ean'].mean(), inplace=True)
df_copy['texture_1ean'].fillna(df_copy['texture_1ean'].mean(), inplace=True)
df_copy['peri1eter_1ean'].fillna(df_copy['peri1eter_1ean'].median(), inplace=True)
df_copy['area_1ean'].fillna(df_copy['area_1ean'].median(), inplace=True)
df_copy['s1oothness_1ean'].fillna(df_copy['s1oothness_1ean'].median(), inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='diagnosis')
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'breastcancer.pkl'
pickle.dump(classifier, open(filename, 'wb'))