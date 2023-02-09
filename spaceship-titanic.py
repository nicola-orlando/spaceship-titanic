#!python
import numpy as np
import pandas as pd
from pandas import read_csv

# Basic plots
import matplotlib.pyplot as plt
import seaborn as sn

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

def conf_matrix_plot(matrix, plot_name):
    fig = plt.figure(figsize = (10, 10))
    plt.title('Correlation matrix')
    ax = sn.heatmap(matrix, cmap='Oranges', annot=True)
    plt.tight_layout()
    plt.savefig('heatmap_'+plot_name+'.png')

data = read_csv('spaceship-titanic/train.csv')
data_test = read_csv('spaceship-titanic/test.csv')
data_sub = read_csv('spaceship-titanic/test.csv')

print(data.head(5))

# Analysis of the NaNs

# Count how many nans there are in each column
for column in data.columns:
    num_of_nans = data[column].isna().sum()
    print('\n\n Column '+str(column)+' has '+str(num_of_nans)+' nans')
    print(data.groupby(data[column].isnull()).mean(numeric_only=True))



import sys
sys.exit()





    
# Look at the input data
print('Dataframe shape')
print(data.shape)

# Select the columns you want to use
# PassengerId HomePlanet CryoSleep  Cabin  Destination   Age    VIP  RoomService  FoodCourt  ShoppingMall     Spa  VRDeck               Name  Transported
# data = data[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'VRDeck', 'Transported']]

print('Total length of the dataset before nans removal = '+str(len(data)))
data = data.dropna()
print('Total length of the dataset after nans removal = '+str(len(data)))

data = data.drop(['Name', 'PassengerId'], axis = 1)
data_test = data_test.drop(['Name', 'PassengerId'], axis = 1)

# Cathegorical data 
data['VIP'] = data['VIP'].map({False: 0, True: 1})
data['CryoSleep'] = data['CryoSleep'].map({False: 0, True: 1})
data['Transported'] = data['Transported'].map({False: 0, True: 1})
data_test['VIP'] = data_test['VIP'].map({False: 0, True: 1})
data_test['CryoSleep'] = data_test['CryoSleep'].map({False: 0, True: 1})

# Convert other categorical data objects into numbers
categorical_data = ['HomePlanet','Cabin','Destination']
for cat_data_to_cnv in categorical_data :
  print ("Handling now data category "+cat_data_to_cnv)
  data[cat_data_to_cnv] = pd.Categorical(data[cat_data_to_cnv]).codes
  data_test[cat_data_to_cnv] = pd.Categorical(data_test[cat_data_to_cnv]).codes

print(data.head(5))
print(data_test.head(5))

# Variables to drop
variables_to_drop = ['Cabin', 'Age', 'VIP', 'FoodCourt', 'ShoppingMall', 'RoomService', 'Spa'] 
for variable in variables_to_drop:
    data = data.drop(variable, axis=1)
    data_test = data_test.drop(variable, axis=1)

# print(data.corr())
conf_matrix_plot(data.corr(), 'correlation_features')

# Look at the input data
print('Dataframe shape')
print(data.shape)












#X_train = data.drop('Transported', axis = 1)
#y_train = data['Transported']
#X_test = data_test

X = data.drop('Transported', axis = 1)
y = data['Transported'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
#predictions = [True if value== 1 else False for value in y_pred]
predictions = [value for value in y_pred]
#print(predictions)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#data_submission = data_sub
#data_submission['Transported']=pd.Series(predictions)
#data_submission = data_submission[['PassengerId','Transported']]
#print(data_submission.head())
#data_submission.to_csv('submission_1.csv', index=False)
