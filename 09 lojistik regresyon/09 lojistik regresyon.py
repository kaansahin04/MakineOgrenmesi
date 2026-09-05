import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veriler=pd.read_csv("eksikveriler.csv")

#impute
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
eksikler=veriler.iloc[:,3:4].values
eksikler=imputer.fit_transform(eksikler)
veriler.iloc[:,3:4]=eksikler

#encode
ulke=veriler.iloc[:,0:1].values
cinsiyet=veriler.iloc[:,-1:].values
lbe=LabelEncoder()
ulke[:,0]=lbe.fit_transform(ulke.ravel())
cinsiyet[:,0]=lbe.fit_transform(cinsiyet.ravel())
ohe=OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
cinsiyet=ohe.fit_transform(cinsiyet).toarray()

#concat
ulke=pd.DataFrame(data=ulke,index=range(len(veriler)))
ozellikler=pd.concat([ulke.iloc[:,1:],veriler.iloc[:,1:4]],axis=1)
hedef=pd.DataFrame(data=cinsiyet[:,1:],index=range(len(veriler)))

#split
x_train,x_test,y_train,y_test=train_test_split(ozellikler.values,hedef.values,test_size=0.33,random_state=0)

#scale
xScaler=StandardScaler()
xScaledTr=xScaler.fit_transform(x_train)
xScaledTe=xScaler.transform(x_test)

#LOJİSTİK REGRESYON

from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(random_state=0)
lgr.fit(xScaledTr, y_train.ravel())
y_pred=lgr.predict(xScaledTe)

#KARMAŞIKLIK MATRİSİ

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

#ÇEŞİTLİ METRİKLER

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(acc)
print(prec)
print(rec)
print(f1)