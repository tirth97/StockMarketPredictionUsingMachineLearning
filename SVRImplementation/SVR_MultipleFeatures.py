import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
dates=[]
features = []
prices = []
month={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

def organize_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        i = 0
        for row in csvFileReader:
            r = []
            a=row[0].split('-')
            #dates.append(int(a[0]) + int(month[a[1]])*100 + (2000+int(a[2]))*10000)
            r.append(float(row[1]))
            r.append(float(row[2]))
            r.append(float(row[3]))
            r.append(long(row[5]))
            r.append(float(row[6]))
            prices.append(float(row[4]))
            features.append(r)
            
            
        return

def predict_prices(features,prices,x):
    #dates = np.reshape(dates,(len(dates),1))
 #   features =np.reshape(features,len(features),len(features[0]))
    svr_lin=SVR(kernel='poly', C=1e3)
    #print(len(features), len(prices))
    splitpoint=int((0.8)*len(features))
    training=features[:splitpoint]
    test=features[splitpoint:]

    svr_lin.fit(training,prices[:splitpoint])
    
    #svrPredict=svr_lin.predict(x)
    #plt.scatter(dates,svr_lin.fit(features, prices),color='red',label='Linear Model')
    #return svrPredict[0]
    return svr_lin.predict(test)[0]

    '''    plt.scatter(features,prices, color='black', label='Data')
    plt.xlabel('Features')
    plt.ylabel('Price')

    plt.legend()

    plt.show()'''

organize_data('applLatest.csv')

arr=[{147.54,147.69,146.98,147.67,10439265}]

predictedPrice=predict_prices(features,prices, arr)
print(predictedPrice)
