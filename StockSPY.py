#Import libraries
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing_extensions import dataclass_transform
import sys
import matplotlib.dates as mdates


#Load the data
def loadData():
    data = pd.read_csv('SPY.csv')
    return data


#Create and train the model
def makeAndTrainModel(trainData, features, target):
    model = xgb.XGBRegressor()#Use regressor
    model.fit(trainData[features], trainData[target])#this trains the model
    return model



def getAccuracy(testData, features, target, model):
    #Show the models accuracy
    accuracy = model.score(testData[features], testData[target])
    print('Accuracy')
    print(accuracy)


def makeDateTimeFormatData(data, testData):

    #Make the date column from the csv into datetime format
    data['Date'] = pd.to_datetime(data['Date'])

def makeDateTimeFormatTestData(data, testData):
    #Make the date column from the test data into datetime format
    testData.loc[:, 'Date'] = pd.to_datetime(testData['Date'])


def plotData(data, testData, predictions):

    #Plot actual and predicted data
    plt.plot(data['Date'], data['Close'], label='Close Price', linestyle='-', color='blue')
    plt.plot(testData['Date'], predictions, label='Predictions', linestyle='--', color='orange')

    #Plot x and y labels
    plt.xlabel('Date')
    plt.ylabel('Price ($)')

    #Proper orientation for x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    plt.legend()
    plt.show()


def main():

    data = loadData()

    #Split the data into training and testing data sets
    trainData = data.iloc[:int(.9*len(data)), :]
    testData = data.iloc[int(.9*len(data)):, :]

    #Define the features and target variable
    #Use opening price and trading volume as data to train
    #We are trying to predict the closing value

    features = ['Open', 'Volume']
    target = 'Close'

    #make the model
    model = makeAndTrainModel(trainData, features, target)

    #Make and show the predictions on test data
    predictions = model.predict(testData[features])

    #Get the accuracy of our model
    getAccuracy(testData, features, target, model)


    #Put date in proper datetime format
    makeDateTimeFormatData(data, testData)
    makeDateTimeFormatTestData(data, testData)

    #plot the data
    plotData(data, testData, predictions)

    sys.exit()


if __name__ == "__main__":
    main()