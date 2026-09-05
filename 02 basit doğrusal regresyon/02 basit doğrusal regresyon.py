import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("satis.csv")
print(veriler)
aylar=veriler[["Aylar"]]
satislar=veriler[["Satislar"]]

#VERİ KÜMESİNİ AYIRMA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)
"""
#ÖZNİTELİK ÖLÇEKLEME (SCALE)
from sklearn.preprocessing import StandardScaler
stScaler=StandardScaler()
X_train=stScaler.fit_transform(x_train)
X_test=stScaler.transform(x_test)
Y_train=stScaler.fit_transform(y_train)
Y_test=stScaler.transform(y_test)
"""
#MODELLEME

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)    #x_train'den y_train'i öğren
tahmin = lr.predict(x_test)      #x_test'ten tahmin üret (y_test ile ne kadar aynı?)

#GÖRSELLEŞTİRME

xtrainSıra = x_train.sort_index()      #random_state ile rastgele ayrılan veriler index'e göre sıralanır
ytrainSıra = y_train.sort_index()
plt.plot(xtrainSıra, ytrainSıra)    #grafik getir
plt.plot(x_test, tahmin)
plt.title("Aylara Göre Satışlar")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")