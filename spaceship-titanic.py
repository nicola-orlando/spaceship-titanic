#!python
import numpy as np
import pandas as pd
from pandas import read_csv

# Basic plots
import matplotlib.pyplot as plt
import seaborn as sn

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

# General options
drop_nans = True 
fill_nans_with_average = False 
remove_outliers = False
variables_to_drop_training = ['PassengerId', 'Num']
variables_to_drop_training = ['Cabin', 'PassengerId', 'Num', 'VIP', 'ShoppingMall', 'Group'] # Set 2, 78.52%
variables_to_drop_training = ['Cabin', 'PassengerId', 'Num', 'VIP', 'Group'] # Set 3, 79.50%
variables_to_drop_training = ['Cabin', 'PassengerId', 'Num'] # Set 1, 80.41%
dummy_optimisation = False
feature_importance_plot = False 

def conf_matrix_plot(matrix, plot_name):
    fig = plt.figure(figsize = (10, 10))
    plt.title('Correlation matrix')
    mask = np.triu(matrix)
    ax = sn.heatmap(matrix, annot=True, fmt='.1f', vmin=-1, vmax=1, center= 0, cmap= 'vlag') # mask=mask)
    plt.tight_layout()
    plt.savefig('heatmap_'+plot_name+'.png')    
    
# Data to be manipulated 
data = read_csv('spaceship-titanic/train.csv')
# Data, just to be used for plotting, e.g. categorical data is unconverted 
data_orig = read_csv('spaceship-titanic/train.csv')
# Data for testing 
data_test = read_csv('spaceship-titanic/test.csv')
# Data for submission
data_sub = read_csv('spaceship-titanic/test.csv')

# ------------------------------------------- Analysis of the NaNs

# Count how many nans there are in each column
print('\n Looking at distribution of NaNs')
for column in data.columns:
    num_of_nans = data[column].isna().sum()
    print('\n\n Column '+str(column)+' has '+str(num_of_nans)+' nans')
    print(data.groupby(data[column].isnull()).mean(numeric_only=True))

# Count how many nans there are in each column
print('\n Looking at distribution of NaNs in the test dataset')
for column in data_test.columns:
    num_of_nans = data_test[column].isna().sum()
    print('\n\n In data_test column '+str(column)+' has '+str(num_of_nans)+' nans')
    print(data_test.groupby(data_test[column].isnull()).mean(numeric_only=True))
    
if drop_nans: 
    data = data.dropna()
    data_orig = data_orig.dropna()

# ------------------------------------------- Features analysis

# Starting with Cabin, Cabin is defined as deck/num/side, define then 3 new columns in the dataframe based on these 

from math import nan

data_orig['Deck'] = data_orig['Cabin'].apply(lambda x: x.split('/')[0] if(x==x) else nan)
data_orig['Num'] = data_orig['Cabin'].apply(lambda x: x.split('/')[1] if(x==x) else nan)
data_orig['Side'] = data_orig['Cabin'].apply(lambda x: x.split('/')[2] if(x==x) else nan)
data['Deck'] = data['Cabin'].apply(lambda x: x.split('/')[0] if(x==x) else nan)
data['Num'] = data['Cabin'].apply(lambda x: int(x.split('/')[1]) if(x==x) else nan)
data['Side'] = data['Cabin'].apply(lambda x: x.split('/')[2] if(x==x) else nan)

data_test['Deck'] = data_test['Cabin'].apply( lambda x: x.split('/')[0] if(x==x) else nan)
data_test['Num'] = data_test['Cabin'].apply( lambda x: int(x.split('/')[1]) if(x==x) else nan)
data_test['Side'] = data_test['Cabin'].apply( lambda x: x.split('/')[2] if(x==x) else nan)

# Now look at PassengerId, decompose this into two parts, a group code and group size (see definition of PassengerId)
data['Group'] = data['PassengerId'].apply(lambda x: int(x.split('_')[0]) if(x==x) else nan)
data['Group_size'] = data['PassengerId'].apply(lambda x: int(x.split('_')[1]) if(x==x) else nan)
data_test['Group'] = data_test['PassengerId'].apply(lambda x: int(x.split('_')[0]) if(x==x) else nan)
data_test['Group_size'] = data_test['PassengerId'].apply(lambda x: int(x.split('_')[1]) if(x==x) else nan)

# Cathegorical data 
data['VIP'] = data['VIP'].map({False: 0, True: 1})
data['CryoSleep'] = data['CryoSleep'].map({False: 0, True: 1})
data['Transported'] = data['Transported'].map({False: 0, True: 1})
data_test['VIP'] = data_test['VIP'].map({False: 0, True: 1})
data_test['CryoSleep'] = data_test['CryoSleep'].map({False: 0, True: 1})

# Convert other categorical data objects into numbers
categorical_data = ['HomePlanet','Cabin','Destination', 'PassengerId']
for cat_data_to_cnv in categorical_data :
  print ("Handling now data category "+cat_data_to_cnv)
  data[cat_data_to_cnv] = pd.Categorical(data[cat_data_to_cnv]).codes
  data_test[cat_data_to_cnv] = pd.Categorical(data_test[cat_data_to_cnv]).codes

if fill_nans_with_average: 
    data=data.fillna(data.mean())
    data_orig=data_orig.fillna(data_orig.mean())

# ------------------------------------------- Detect and remove outliers

def detect_outlier(feature):
    outliers = []
    data_feat = data[feature]
    mean = np.mean(data_feat)
    std =np.std(data_feat)
    
    
    for y in data_feat:
        z_score= (y - mean)/std 
        if np.abs(z_score) > 3:
            outliers.append(y)
    print('\nOutlier caps for {}:'.format(feature))
    print('  --95p: {:.1f} / {} values exceed that'.format(data_feat.quantile(.95),
                                                             len([i for i in data_feat
                                                                  if i > data_feat.quantile(.95)])))
    print('  --3sd: {:.1f} / {} values exceed that'.format(mean + 3*(std), len(outliers)))
    print('  --99p: {:.1f} / {} values exceed that'.format(data_feat.quantile(.99),
                                                           len([i for i in data_feat
                                                                if i > data_feat.quantile(.99)])))

# Determine what the upperbound should be for continuous features
if remove_outliers: 
    for feat in ['Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Group','Group_size']:
        detect_outlier(feat)
        data[feat] = data[feat].clip(upper=data[feat].quantile(.99))
    
# ------------------------------------------- Further analyse categorical data 
 
# Plot correlation between the inputs, all numberical variables or categorical converted to numerical 
variables_to_drop = ['Name'] 
for variable in variables_to_drop:
    data = data.drop(variable, axis=1)
    data_test = data_test.drop(variable, axis=1)
conf_matrix_plot(data.corr(), 'correlation_features')

# Look at unique values of the categorical data
all_cat_data = ['HomePlanet','Cabin','Destination', 'PassengerId', 'VIP', 'CryoSleep']
for element in all_cat_data:
    print('Unique values for categorical data '+str(element)+' ='+str(data[element].nunique()))

# Look now at some tables to illustrate how each variable changes depending on the target variable 
# Start with categorical data with low multiplicity of unique values: HomePlanet, Destination, VIP, CryoSleep
simple_cat_data = ['HomePlanet','Destination','VIP', 'CryoSleep']
for element in simple_cat_data:
    print('\n Looking at categorical data '+str(element)+'\n')
    print(data_orig[[element,'Transported']].groupby(element).mean())

# Convert categorical Deck/Side data objects into numbers so we canlcualte their correlation and use it in the training
categorical_data_cabin = ['Deck','Side']
for cat_data_to_cnv in categorical_data_cabin:
  print ("Handling now data category "+cat_data_to_cnv)
  data[cat_data_to_cnv] = pd.Categorical(data[cat_data_to_cnv]).codes
  data_test[cat_data_to_cnv] = pd.Categorical(data_test[cat_data_to_cnv]).codes

# Look at new categorical data defined from Cabin   
decomposed_cabin = ['Deck', 'Num', 'Side']
for element in decomposed_cabin:
    print('\n Looking at categorical data '+str(element)+'\n')
    print(data_orig[[element,'Transported']].groupby(element).mean())
    print('Now looking at the correlation with the target label')
    print(data[element].corr(data['Transported']))
    
print('\n Looking at Group_size column and average values of the target label\n')
print(data[['Group_size','Transported']].groupby('Group_size').mean())

# Let's add one combined feature summing up those among the RoomService, FoodCourt, ShoppingMall, Spa, VRDeck features with same sign correlation,
# that would be RoomService, Spa, VRDeck
#data['Goods_comb'] = data['RoomService']+data['Spa']+data['VRDeck']
#data_test['Goods_comb'] = data_test['RoomService']+data_test['Spa']+data_test['VRDeck']
    
# ------------------------------------------- Final selection of the training features 

# Variables to drop
for variable in variables_to_drop_training:
    data = data.drop(variable, axis=1)
    data_test = data_test.drop(variable, axis=1)

# ------------------------------------------- Training with 5-fold cross validation 
    
X = data.drop('Transported', axis = 1)
y = data['Transported'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


model = XGBClassifier()

from sklearn.model_selection import GridSearchCV

parameters = {}

if dummy_optimisation:
    parameters = {
        'max_depth': [5,6,7] # Default is 6 
    }
else:
    parameters = {
        'eta': [0.05*i for i in range(1, 10)], # Default is 0.3
        'gamma': [0.05*i for i in range(0, 3)], # Default is 0 
        'max_depth': [4,5,6,7,8], # Default is 6 
        'min_child_weight': [1,2,5,7,8], # Default is 1
        'subsample': [1,0.8,0.5], # Default is 1 
        'lambda': [0.2*i for i in range(0, 6)], # Default is 1
        'alpha': [0.1*i for i in range(0, 6)], # Default is 0
        'max_leaves': [0,1,2] # Default is 0 
    }    
    
cv = GridSearchCV(model, parameters, cv=5)
cv.fit(X_train, y_train)


if feature_importance_plot: 
    fig = plt.figure(figsize = (4, 8))
    feat_imp = model.feature_importances_
    indices = np.argsort(feat_imp)
    plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
    plt.barh(range(len(indices)), feat_imp[indices], color='r', align='center')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

y_pred = cv.best_estimator_.predict(X_test)
predictions = [value for value in y_pred]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Now calculate prediction on the full actual test set
y_pred_final = cv.best_estimator_.predict(data_test)
predictions = [True if value== 1 else False for value in y_pred_final]
data_submission = data_sub
data_submission['Transported']=pd.Series(predictions)
data_submission = data_submission[['PassengerId','Transported']]
data_submission.to_csv('submission_1.csv', index=False)
