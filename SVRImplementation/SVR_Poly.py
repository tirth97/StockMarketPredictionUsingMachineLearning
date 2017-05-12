import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def organize_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(float(row[1]))
            prices.append(float(row[4]))
        return

def predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))

    svr_poly=SVR(kernel='poly', C=0.02, degree=2)

    svr_poly.fit(dates,prices)

    plt.scatter(dates,prices, color='black', label='Data')
    plt.plot(dates,svr_poly.predict(dates),color='red',label='Polynomial Model')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.legend()

    plt.show()

    return svr_poly.predict(x)[0]

organize_data('applLatest.csv')

predictedPrice=predict_prices(dates,prices, 147.54)

print(predictedPrice)
