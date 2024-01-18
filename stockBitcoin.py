import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from typing_extensions import dataclass_transform
#Load the data

data = pd.read_csv('bitcoin.csv')
#data
#show the data visually
data['Close'].plot()

#Split the data into training and testing data sets

trainData = data.iloc[:int(.8*len(data)), :]
testData = data.iloc[int(.8*len(data)):, :]

#Define the features and target variable

features = ['Open', 'Volume']
target = 'Close'



#Create and train the model
model = xgb.XGBRegressor()
model.fit(trainData[features], trainData[target])#this trains the model


#Make and show the predictions on test data

predictions = model.predict(testData[features])
print('Model Predictions')
print(predictions)


#Show the actual values
print('Actual values')
print(testData[target])




#Show the models accuracy
accuracy = model.score(testData[features], testData[target])
print('Accuracy')
print(accuracy)



#plot the predictions and the close price
plt.plot(data['Close'], label = 'Close Price')
plt.plot(testData[target].index, predictions, label = 'Predictions')
plt.legend()
plt.show()