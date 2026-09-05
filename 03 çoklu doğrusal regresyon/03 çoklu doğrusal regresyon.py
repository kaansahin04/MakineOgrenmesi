import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("veriler.csv")
print(veriler)      #verileri yazdır

#KATEGORİK VERİLERİ SAYISAL VERİLERE DÖNÜŞTÜRME (ENCODE)
from sklearn import preprocessing
ülke = veriler.iloc[:,0:1].values
labelEncoding = preprocessing.LabelEncoder()
ülke[:,0] = labelEncoding.fit_transform(veriler.iloc[:,0])
print(ülke)
oneHotEncoding = preprocessing.OneHotEncoder()
ülke = oneHotEncoding.fit_transform(ülke).toarray()
print(ülke)
cinsiyet = veriler.iloc[:,-1:].values
labelEncoding2 = preprocessing.LabelEncoder()
cinsiyet[:,-1] = labelEncoding2.fit_transform(veriler.iloc[:,-1])
print(cinsiyet)
oneHotEncoding2 = preprocessing.OneHotEncoder()
cinsiyet = oneHotEncoding2.fit_transform(cinsiyet).toarray()
print(cinsiyet)

#DATAFRAME OLUŞTURMA VE BİRLEŞTİRME
ülkeler=pd.DataFrame(data=ülke, index=range(len(veriler)), columns=["fr","tr","us"])
sayılar=pd.DataFrame(data=veriler.iloc[:,1:4].values, index=range(len(veriler)), columns=["boy","kilo","yaş"])
cinsiyetler=pd.DataFrame(data=cinsiyet[:,:1], index=range(len(veriler)), columns=["cinsiyet"])
bilgiler=pd.concat([ülkeler,sayılar], axis=1)
herşey=pd.concat([bilgiler,cinsiyetler], axis=1)

#VERİ KÜMESİNİ AYIRMA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(bilgiler,cinsiyetler,test_size=0.33,random_state=0)
"""
#ÖZNİTELİK ÖLÇEKLEME (SCALE)
from sklearn.preprocessing import StandardScaler
stScaler=StandardScaler()
X_train=stScaler.fit_transform(x_train)
X_test=stScaler.transform(x_test)
"""
#MODELLEME
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)

#BOY TAHMİNİ İÇİN BOY SÜTUNU AYRI, KALAN SÜTUNLAR AYRI ALINIYOR

boy=herşey.iloc[:,3:4].values
print(boy)
kalan1=herşey.iloc[:,:3]
kalan2=herşey.iloc[:,4:]
kalan=pd.concat([kalan1,kalan2], axis=1)
print(kalan)
x_train2,x_test2,y_train2,y_test2=train_test_split(kalan,boy,test_size=0.33,random_state=0)
lr2=LinearRegression()
lr2.fit(x_train2,y_train2)
tahmin2=lr2.predict(x_test2)

#GERİ ELEME (HER SÜTUNU ALMAK DOĞRU OLMAYABİLİR, HANGİLERİNİN ALINMASI GEREKTİĞİNİN KARARININ YÖNTEMİ)

import statsmodels.api as sm

yeniKalan = sm.add_constant(kalan)      #tamamen 1'ler içeren yeni sütun oluşturulur. (sabit değeri için)
yeniKalan=np.array(yeniKalan, dtype=float)
model = sm.OLS(boy, yeniKalan).fit()
print(model.summary())
#en yüksek p değeri index=4 sütununda olduğundan elenir (genelde p sınırı 0.05 olarak alınır)
yeniKalan=yeniKalan[:,[0,1,2,3,4,6]]
yeniKalan=np.array(yeniKalan, dtype=float)
model = sm.OLS(boy, yeniKalan).fit()      #her sütunun boy sütunu üzerindeki etkisi
print(model.summary())
#yüksek p değerli sütunun çıkarılması sonrası yeni tahminler
x_train3,x_test3,y_train3,y_test3=train_test_split(yeniKalan,boy,test_size=0.33,random_state=0)
lr3=LinearRegression()
lr3.fit(x_train3,y_train3)
tahmin3=lr3.predict(x_test3)