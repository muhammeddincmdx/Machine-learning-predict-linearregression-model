#Linear regression


#Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#reading data

veri = pd.read_csv("satislar.csv")
print(veri)

#Seperating data

aylar = veri[["Aylar"]]
satislar = veri[["Satislar"]]

satislar2 = veri.iloc[:,:1]


#splitting for train and test

from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(aylar,satislar,test_size = 0.33, random_state = 0)
'''
# scaling of data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


tahmin = lr.predict(x_test)  # what we want to predict we give x test and the system find Y_test


#making graph for analysis
x_train = x_train.sort_index() #sort xtrain like 1,2,..
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))


plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satış")