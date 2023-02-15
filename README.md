# spaceship-titanic
Repository for the spaceship Titanic challenge: https://www.kaggle.com/competitions/spaceship-titanic/overview.    

Here also some notes on the analysis and results.  

## EDA of the dataset 

A detailed description of the dataset is available here https://www.kaggle.com/competitions/spaceship-titanic/data. 

One can check the overall features of the dataset: 
- The training data has a balanced set of labels 
- There are some columns with outliers (for example RoomService has mean 223, std dev. of 645 and max value of 9920). Outliers will be handled separately

The target label is Transported. To have a quick idea of how the features are related to the target label here I add a correlation plot. 
This plot is made right after converting to numerical all categorical data and dropping the NaNs. 

![heatmap_correlation_features](https://user-images.githubusercontent.com/26884030/217801029-a01df567-f361-45a6-8120-e8d12645f60e.png)

From this plot, one might infer the following: 
- CryoSleep, RoomService, Spa, VRDeck are the features with most correlation with the target features 
- PassengerId, ShoppingMall, VIP are the less correlated, thus might be less important for the model design

### Analysis of individual features 

This study is done after removing any row with at least one NaN in the dataset. 
The first thing I want to do is to check how many unique values each categorical data feature has. Here's the result 

['HomePlanet','Cabin','Destination', 'PassengerId', 'VIP', 'CryoSleep']

HomePlanet | Cabin | Destination | PassengerId | VIP | CryoSleep 
--- | --- | --- |--- |--- |--- 
3 | 5305 | 3 | 6606 | 2 | 2 

As expected Cabin and PassengerId are those categorical data with highest number of unique values. Is this useful/good for the performance of the model? Likely not, so I will further process them. 

Looking at categorical data with low multiplicity of distinct counts, one can see the following probability of being 'transported' (Transported label = True), assuming fully balanced dataset (practically the case) 

- HomePlanet: people from Europa are significantly more likely to be transported, people from Earth are less likely transported
- Destination: people directed to '55 Cancri e' are significantly more likely to be transported 
- VIP: look like VIPs are less likely to be transported 
- CryoSleep: people in CryoSleep are strongly likely to be transported 

Summary tables here below. The second column is the average value of the target label (Transported)

HomePlanet | Transported label average
--- | --- 
Earth | 0.43 
Europa |   0.66
Mars | 0.52
  
Destination | Transported label average
--- | --- 
55 Cancri e | 0.62 
PSO J318.5-22 | 0.51 
TRAPPIST-1e | 0.47 

VIP | Transported label average
--- | --- 
False | 0.51 
True |  0.37

CryoSleep | Transported label average
--- | --- 
False | 0.33 
True | 0.82 


In summary using these four features (HomePlanet, Destination, VIP, CryoSleep) is likely to be beneficial. 

Now analysing the categorical data with high multiplicity of unique values (Cabin and PassengerId). To do so we will go back to the dataset before converting the categorical data to numerical. 

Starting with Cabin. This has -0.1 correlation with the target label. The variables is defined in this way: "The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard". It is beneficial to see the impact of decomposing the Cabin variable into each component, deck/num/side. Here the results 

- Deck: this seems to have a very important role, passengers from decks B and C are very likely to be transported (probability of about 70%)
- Side: passengers in side S are much more likely (20% more) to be transported

Now let's calculate the correlation between the target label and the new 3 columns, the results are: 

- Deck: -0.11
- Side: 0.10
- Number: -0.05

The results are pretty interesting, Deck and Side are anti-correlated so merging these two features as part of Cabin can potentially result in loss of information and ability to distinguish between the two Transported outcomes.  

Finally let's look at PassengerId, its definition is "A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.". As for the Cabin variable let's split this into two components: "gggg" (Group) and "pp" (Group_size). A reminder, the PassengerId variable has correlation below 0.005 with the target. What do we have now?
- Group: has no correlation with the target and very large correlation with the cabin number !
- Group_size has significant correlation with target (10%)

How does the average label value changes per group size? 

Group Size | Transported label average
--- | --- 
1 |             0.48
2 |             0.56
3 |             0.60
4 |             0.64
5 |             0.57
6 |             0.55
7 |             0.59
8 |             0.50

Finally let's try to further process the Names of the passengers looking for some patterns. 

A feature that is possible to extract from the data is the gender of the passenger. 
In the original Titanic (https://www.kaggle.com/competitions/titanic) challenge the gender of the passengers is the single most influential feature.  
Here we can extract it based on the passenger first name. 

Using gender_guesser (https://pypi.org/project/gender-guesser/) you will get the following partition of the train dataset 

Gender | Number of labeled rows 
--- | --- 
andy          | 71
female        | 811
male          | 423
mostly_female | 23
mostly_male   | 81
unknown       | 7084

The average label value changes per gender group is found to be  

Gender | Transported label average
--- | --- 
andy | 0.46
female | 0.47
male | 0.44
mostly_female | 0.52
mostly_male | 0.48
unknown | 0.51

Because of the large fraction of data classified as unknown, the gender information will hardly be useful. 

Alternatives for the gender classification based on first names also fail for multiple reasons: names belong to populations with multiple nationalities, some names are short cuts or nicknames, there are several typos such as swapping of nearby letters, etc..

I finally tried to look for hidden features in the names of the passengers. For example I tried to count the number of vowels and consonants in each name and check if these properties have some degree of separation power for the label to predict. It was not the case. 

### Nans analysis 

Nans counting per column: 

PassengerId | HomePlanet | CryoSleep | Cabin | Destination | Age | VIP | RoomService | FoodCourt | ShoppingMall | Spa | VRDeck | Name | Transported
--- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- 
0 | 201 | 217 | 199 | 182 | 179 | 203 | 181 | 183 | 208 | 183 | 188 | 200 | 0 

Is any set of Nans correlated with the target variable Transported? 
Looking only at numerical data: look like Nans in RoomService, FoodCourt, ShoppingMall have some correlation with the target variable: 
- Average value of the transported label is 0.504582 and 0.458564 for RoomService not having and having a Nan
- Average value of the transported label is 0.502820 and 0.540984 for FoodCourt not having and having a Nan
- Average value of the transported label is 0.502534 and 0.548077 for ShoppingMall not having and having a Nan

Two approaches for nans replacements will be used

1. Remove any row with a nan
2. Within a column replace the nans with with average values across the given column
3. For RoomService, FoodCourt, ShoppingMall, Spa, VRDeck only. For these variables a nan is really treated as a missing entry, that is like a zero (as it would mean that a passenger didn't spend credit on goods on the spaceship).  

### Outliers 

Outliers are detected by looking for data not belonging to the 99% quantiles for the rescepctive sample distributions they are extracted from. 
Outliers are checked for the features RoomService, FoodCourt, ShoppingMall, Spa, VRDeck (and Age). 

### Final model 

The model I decided to use is XGBoost implemented with a 5-fold cross validation strategy. I tried several configurations to test variations of input features, nan treatment, XGBoost model optimisation, etc. I calculated the classification accuracy with a small testing sample obtained by reserving 20% of the initial training dataset. For convenience I define a nominal model as follows: 
1. Training features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Deck', 'Side', 'Group', 'Group_size', 'Gender', '#Vowels', '#Consonant']
2. Nans in numerical variables ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'] are replaced with zeros. 
3. Rows with nans in the training features are removed
4. Outliers are kept 
5. Only a simple optimisation is performed on the for the max_depth parameter of the XGBoost model ('max_depth': [5,6,7])

The results are as shown below.  

Configuration | Classification accuracy [%] 
--- | --- 
Model 1: Nominal |             81.08
Model 2: Nominal, replace nans with averages (features other than in point 2) |             81.08
Model 3: Nominal + Outliers dropped |             80.50
Model 4: Nominal + reduced features set 1 (1)  |             80.18
Model 5: Nominal + reduced features set 2 (2) |             80.18
Model 6: Nominal + reduced features set 3 (3)|             80.68
Model 7: Nominal + reduced features set 2 + HP scan |   79.92           
Model 8: Nominal + reduced features set 3 + HP scan |   80.80

1. Set 1: ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Deck', 'Side', 'Group_size', 'Gender', 'Vowels', 'Consonant']

2. Set 2: ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Deck', 'Side', 'Group_size']

3. Set 3: ['HomePlanet', 'CryoSleep', 'Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Deck', 'Side', 'Group_size']

The last two columns of the table above include two training configurations with a a detailed optimisation of the parameters of the model. 
The optimisation is performed by means of a grid search with the following options 

- 'eta': [0.05*i for i in range(2, 8)]
- 'gamma': [0.05*i for i in range(0, 3)]
- 'max_depth': [4,5,6,7,8]
- 'max_leaves': [0,1,2]

See the XGBoost manual for the definition of these parameters (https://xgboost.readthedocs.io/en/stable/).

Here below and example of features ranking following the training of Model 8 (see the table above)

![feature_importance](https://user-images.githubusercontent.com/26884030/219045488-f73b929a-8360-4b40-8627-11a80854add1.png)

### Results 

For the submission in Kaggle used Model 8, refitted to the full training data. The final score on the Kaggle submission file is 0.80289, top 20% of the leaderboard. 
