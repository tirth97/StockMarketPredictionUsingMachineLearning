import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []
month={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

def organize_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            a=row[0].split('/')

            dates.append(float(row[1]))
            prices.append(float(row[4]))
        return

def predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))

    svr_rbf=SVR(kernel='rbf', C=1, gamma=0.1)

    svr_rbf.fit(dates,prices)

    plt.scatter(dates,prices, color='black', label='Data')
    plt.plot(dates,svr_rbf.predict(dates),color='red',label='Radial Bias Function Model')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.legend()

    plt.show()

    return svr_rbf.predict(x)[0]

organize_data('applLatest.csv')

predictedPrice=predict_prices(dates,prices, 147.54)

print(predictedPrice)
