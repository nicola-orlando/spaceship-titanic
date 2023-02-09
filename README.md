# spaceship-titanic
Repositiory for the spaceship Titanic challenge: https://www.kaggle.com/competitions/spaceship-titanic/overview.    

Here also some notes on the analysis. 

## EDA of the dataset 
A detailed description of the dataset is available here https://www.kaggle.com/competitions/spaceship-titanic/data. 

The target variable is Transported. To have a quick idea of how the features are related to transported here a correlation plot. 
This plot is made right after converting to numerical all categorical data and dropping the NaNs. 

![heatmap_correlation_features](https://user-images.githubusercontent.com/26884030/217801029-a01df567-f361-45a6-8120-e8d12645f60e.png)

From this plot, one might infer the following: 
- CryoSleep, RoomService, Spa, VRDeck are the features with most correlation with the target features 
- PassengerId, ShoppingMall, VIP are the less correlated, thus might be less important for the model design (is this really true?)

### Analysis of individual features 

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
