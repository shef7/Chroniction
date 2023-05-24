

# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv(r'C:\Chroniction\Multi_Disease_Predictor\models\lungdata.csv')

# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'LungCancerPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df.copy(deep=True)
df_copy[['Age','Smokes','AreaQ','Alkhol']] = df_copy[['Age','Smokes','AreaQ','Alkhol']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Age'].fillna(df_copy['Age'].mean(), inplace=True)
df_copy['Smokes'].fillna(df_copy['Smokes'].mean(), inplace=True)
df_copy['AreaQ'].fillna(df_copy['AreaQ'].median(), inplace=True)
df_copy['Alkhol'].fillna(df_copy['Alkhol'].median(), inplace=True)


# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='Result')
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'lungcancer.pkl'
pickle.dump(classifier, open(filename, 'wb'))