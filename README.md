# spaceship-titanic
Repositiory for the spaceship Titanic challenge: https://www.kaggle.com/competitions/spaceship-titanic/overview.    

Here also some notes on the analysis. 

## EDA of the dataset 
A detailed description of the dataset is available here https://www.kaggle.com/competitions/spaceship-titanic/data. 

One can check the overall features of the dataset: 
- The training data has a balanced set of labels 
- There are some columns with outliers (for example RoomService has mean 223, std dev. of 645 and max value of 9920). Outliers will be handled separately

The target variable is Transported. To have a quick idea of how the features are related to transported here a correlation plot. 
This plot is made right after converting to numerical all categorical data and dropping the NaNs. 

![heatmap_correlation_features](https://user-images.githubusercontent.com/26884030/217801029-a01df567-f361-45a6-8120-e8d12645f60e.png)

From this plot, one might infer the following: 
- CryoSleep, RoomService, Spa, VRDeck are the features with most correlation with the target features 
- PassengerId, ShoppingMall, VIP are the less correlated, thus might be less important for the model design (is this really true?)

### Analysis of individual features 

This study is done after removing any row with at least one NaN in the dataset. 
The first thing I want to do is to check how many unique values each categorical data feature has. Here's the result 

['HomePlanet','Cabin','Destination', 'PassengerId', 'VIP', 'CryoSleep']

HomePlanet | Cabin | Destination | PassengerId | VIP | CryoSleep 
--- | --- | --- |--- |--- |--- 
3 | 5305 | 3 | 6606 | 2 | 2 

As expected Cabin and PassengerId are those categorical data with highest number of unique values. Is this useful/good for the performance of the model? 

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

Starting with Cabin. This has -0.1 correlation with the target label. The variables is defined in this way: "The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard". It is beneficial to see. the imact of decomposing the Cabin variable into each component, deck/num/side. Here the results 

- Deck: this seems to have a very important role, passengers from decks B and C are very likely to be transported (probability of about 70%)
- Side: passengers in side S are much more likely (20% more) to be transported

Now let's calculate the correlation between the target label and the new 3 columns, the results are: 

- Deck: -0.11
- Side: 0.10
- Number: -0.05

The results are pretty interesting, Deck and Side are anti-correlated so merging these two features as part of Cabin can potentially result in loss of information and ability to distinguish between the two Tranported outcomes.  

Finally let's look at PassengerId, its definition is "A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.". As for the Cabin variable let's split this into two components: "gggg" (Group) and "pp" (Group_size). A reminder, the PassengerId variable has correlation below 0.005 with the target. What do we have 
- Group: has no correlation with the target and very large correlation with the cabin number !
- Group_size has large correlation with target (10%)

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

### Nans analysis 

Nans counting per column: 

PassengerId | HomePlanet | CryoSleep | Cabin | Destination | Age | VIP | RoomService | FoodCourt | ShoppingMall | Spa | VRDeck | Name | Transported
--- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- 
0 | 201 | 217 | 199 | 182 | 179 | 203 | 181 | 183 | 208 | 183 | 188 | 200 | 0 

Is any set of Nans correlated with the targed variable Transported? 
Looking only at numerical data: look like Nans in RoomService, FoodCourt, ShoppingMall have some correlation with the target variable: 
- Average value of the transported label is 0.504582 and 0.458564 for RoomService not having and having a Nan
- Average value of the transported label is 0.502820 and 0.540984 for FoodCourt not having and having a Nan
- Average value of the transported label is 0.502534 and 0.548077 for ShoppingMall not having and having a Nan
