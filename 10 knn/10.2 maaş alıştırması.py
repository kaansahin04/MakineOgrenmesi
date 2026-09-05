import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

veriler=pd.read_csv("maas.csv", na_values="?")

#IMPUTE
imputer=SimpleImputer(missing_values=np.nan, strategy="most_frequent")
eksikler=veriler.iloc[:,:].values
eksikler=imputer.fit_transform(eksikler)
veriler.iloc[:,:]=eksikler

#ENCODE
work=veriler.iloc[:,1:2].values
evlilik=veriler.iloc[:,5:6].values
meslek=veriler.iloc[:,6:7].values
iliski=veriler.iloc[:,7:8].values
irk=veriler.iloc[:,8:9].values
cinsiyet=veriler.iloc[:,9:10].values
veriler["native-country"] = np.where(veriler["native-country"] == "United-States", 1, 0)
maas=veriler.iloc[:,-1:].values
lbe=LabelEncoder()
work[:,0]=lbe.fit_transform(work.ravel())
evlilik[:,0]=lbe.fit_transform(evlilik.ravel())
meslek[:,0]=lbe.fit_transform(meslek.ravel())
iliski[:,0]=lbe.fit_transform(iliski.ravel())
irk[:,0]=lbe.fit_transform(irk.ravel())
cinsiyet[:,0]=lbe.fit_transform(cinsiyet.ravel())
maas[:,0]=lbe.fit_transform(maas.ravel())
ohe=OneHotEncoder()
work=ohe.fit_transform(work).toarray()
evlilik=ohe.fit_transform(evlilik).toarray()
meslek=ohe.fit_transform(meslek).toarray()
iliski=ohe.fit_transform(iliski).toarray()
irk=ohe.fit_transform(irk).toarray()
cinsiyet=ohe.fit_transform(cinsiyet).toarray()
maas=ohe.fit_transform(maas).toarray()

#CONCAT
work=pd.DataFrame(data=work,index=range(len(veriler)))
evlilik=pd.DataFrame(data=evlilik,index=range(len(veriler)))
meslek=pd.DataFrame(data=meslek,index=range(len(veriler)))
iliski=pd.DataFrame(data=iliski,index=range(len(veriler)))
irk=pd.DataFrame(data=irk,index=range(len(veriler)))
cinsiyet=pd.DataFrame(data=cinsiyet,index=range(len(veriler)))
ozellikler=pd.concat([veriler.iloc[:,0:1],work.iloc[:,1:],veriler.iloc[:,2:3],veriler.iloc[:,4:5],evlilik.iloc[:,1:],
                      meslek.iloc[:,1:],iliski.iloc[:,1:],irk.iloc[:,1:],cinsiyet.iloc[:,1:],veriler.iloc[:,10:14]],
                     axis=1)
maas=pd.DataFrame(data=maas,index=range(len(veriler)))
hedef=pd.concat([maas.iloc[:,1:]])

#SPLIT
x_train,x_test,y_train,y_test=train_test_split(ozellikler.values,hedef.values,test_size=0.33,random_state=0)

#SCALE
xScaler=StandardScaler()
xScTr=xScaler.fit_transform(x_train)
xScTe=xScaler.transform(x_test)

#MODELİN KURULUMU
knn=KNeighborsClassifier(n_neighbors=23,weights="uniform",metric="manhattan")
knn.fit(xScTr,y_train.ravel())
y_pred=knn.predict(xScTe)

#METRİKLER
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
spec=recall_score(y_test,y_pred,pos_label=0)
f1=f1_score(y_test,y_pred)
print("Karmaşıklık Matrisi:")
print(f"{cm}")
print(f"Accuracy (Doğru Tahmin Oranı): {acc}")
print(f"Precision (Pozitif Tahminlerin Doğru Olma Oranı, TP'nin TP+FP'ye Oranı): {prec}")
print(f"Recall/Sensitivity (Pozitifleri Doğru Tahmin Oranı, TP'nin TP+FN'ye Oranı): {rec}")
print(f"Specificity (Negatifleri Doğru Tahmin Oranı, TN'nin FP+TN'ye Oranı): {spec}")
print(f"F1 (Prec-Rec Karışımı): {f1}")