import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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

#grid arama
olasiDegerler={
    "criterion":["gini","entropy"],
    "max_depth":[5,10,15,20,25],
    "min_samples_leaf":[1,5,10,20],
    "min_samples_split":[10,50,100,300],
}

#KARAR AĞACI

from sklearn.tree import DecisionTreeClassifier

gridAra=GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=olasiDegerler, n_jobs=3)
gridAra.fit(x_train,y_train)
print(f"En uygun parametreler: {gridAra.best_params_}")
dtc=gridAra.best_estimator_
y_pred=dtc.predict(x_test)

#DEĞERLENDİRME METRİKLERİ
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
print(f"Conf: {confusion_matrix(y_test, y_pred)}")
print(f"Acc: {accuracy_score(y_test, y_pred)}")
print(f"Prec: {precision_score(y_test, y_pred)}")
print(f"Rec (Sens): {recall_score(y_test, y_pred)}")
print(f"Spec: {recall_score(y_test, y_pred, pos_label=0)}")
print(f"F1: {f1_score(y_test, y_pred)}")