import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("veriler.csv")
print(veriler)      #verileri yazdır
print(veriler[["boy"]])     #sadece boy sütununu yazdır
print(veriler[["boy","kilo"]])    #sadece boy ve kilo sütunlarını yazdır

#EKSİK VERİ DOLDURMA (IMPUTE)

from sklearn.impute import SimpleImputer

eksikVeriler=pd.read_csv("eksikveriler.csv")
print(eksikVeriler)
imputer = SimpleImputer(
    missing_values=np.nan,      #NaN olan değerleri eksik veri olarak ele al
    strategy="mean"     #sütunun ortalamasını al ve eksik verileri doldur
)
sayısal = eksikVeriler.iloc[:,1:4].values    #her satırın 1,2,3 index'li (sayısal) değerini al
imputer = imputer.fit(sayısal[:,1:4])     #yukarıdaki strateji ile ortalamalarını öğren
sayısal[:,1:4] = imputer.transform(sayısal[:,1:4])     #öğrenilen değerler ile NaN'ları doldur
print(sayısal)

#KATEGORİK VERİLERİ SAYISAL VERİLERE DÖNÜŞTÜRME (ENCODE)

from sklearn import preprocessing

ülke = eksikVeriler.iloc[:,0:1].values     #her satırın 0 index'li (ülke) değerini al
labelEncoding = preprocessing.LabelEncoder()      #LabelEncoding yöntemi
ülke[:,0] = labelEncoding.fit_transform(eksikVeriler.iloc[:,0])   #her ülkeye 1-2-0 değeri ata
print(ülke)
oneHotEncoding = preprocessing.OneHotEncoder()    #OneHotEncoding yöntemi
ülke = oneHotEncoding.fit_transform(ülke).toarray()   #o ülkeye 1, diğerlerine 0 ata; dizi kur
print(ülke)

#DATAFRAME OLUŞTURMA VE BİRLEŞTİRME

ülkeler=pd.DataFrame(data=ülke, index=range(len(eksikVeriler)), columns=["fr","tr","us"])
print(ülkeler)
sayılar=pd.DataFrame(data=sayısal, index=range(len(eksikVeriler)), columns=["boy","kilo","yaş"])
print(sayılar)
cinsiyet = eksikVeriler.iloc[:,-1].values
cinsiyetler=pd.DataFrame(data=cinsiyet, index=range(len(eksikVeriler)), columns=["cinsiyet"])
bilgiler=pd.concat([ülkeler,sayılar], axis=1)      #axis=1 ile uyuşan satırlar yan yana yazılır
print(bilgiler)
herşey=pd.concat([bilgiler,cinsiyetler], axis=1)
print(herşey)

#VERİ KÜMESİNİ AYIRMA

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(bilgiler,cinsiyetler,test_size=0.33,random_state=0)

#ÖZNİTELİK ÖLÇEKLEME (SCALE)

from sklearn.preprocessing import StandardScaler

stScaler=StandardScaler()
x_train=stScaler.fit_transform(x_train)
x_test=stScaler.transform(x_test)