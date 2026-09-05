import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

veriler=pd.read_csv("meyve.csv")

#IMPUTE
#gerek duyulmamıştır

#ENCODE
sekil=veriler.iloc[:,1:2].values
renk=veriler.iloc[:,4:5].values
tat=veriler.iloc[:,5:6].values
meyve=veriler.iloc[:,-1:].values
lbe=LabelEncoder()
sekil[:,0]=lbe.fit_transform(sekil.ravel())
renk[:,0]=lbe.fit_transform(renk.ravel())
tat[:,0]=lbe.fit_transform(tat.ravel())
meyve=lbe.fit_transform(meyve.ravel()).astype(int)
ohe=OneHotEncoder()
sekil=ohe.fit_transform(sekil).toarray()
renk=ohe.fit_transform(renk).toarray()
tat=ohe.fit_transform(tat).toarray()

#CONCAT
sekil=pd.DataFrame(data=sekil,index=range(len(veriler)))
renk=pd.DataFrame(data=renk,index=range(len(veriler)))
tat=pd.DataFrame(data=tat,index=range(len(veriler)))
meyve=pd.DataFrame(data=meyve,index=range(len(veriler)))
ozellikler=pd.concat([veriler.iloc[:,0:1],sekil.iloc[:,1:],veriler.iloc[:,2:4],renk.iloc[:,1:],tat.iloc[:,1:]],
                     axis=1)
hedef=pd.concat([meyve.iloc[:,:]])

#SPLIT
x_train,x_test,y_train,y_test=train_test_split(ozellikler.values,hedef.values,test_size=0.33,random_state=0)

#GRID ARAMA
olasiDegerler={
    "n_estimators":[50,100,150],
    "criterion":["gini","entropy"],
    "max_depth":[5,8,10,13],
    "min_samples_leaf":[30,50,75,100],
    "min_samples_split":[100,150,200]
}
gridAra=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=olasiDegerler,n_jobs=-1)
gridAra.fit(x_train,y_train.ravel())
print(f"En iyi parametreler: {gridAra.best_params_}")

#MODELİN KURULUMU
dtc=gridAra.best_estimator_
y_pred=dtc.predict(x_test)

#METRİKLER
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
prec=precision_score(y_test,y_pred,average="weighted")
rec=recall_score(y_test,y_pred,average="weighted")
f1=f1_score(y_test,y_pred,average="weighted")
print("Karışıklık Matrisi:")
print(f"{cm}")
print(f"Accuracy (Genel Doğru Tahmin Oranı): {acc}")
print(f"Precision (Pozitif Tahmin Edilenlerin Doğruluk Oranı): {prec}")
print(f"Recall/Sensitivity (Pozitif Olanları Doğru Tahmin Oranı): {rec}")
print(f"F1 (Prec-Rec Karışımı): {f1}")
