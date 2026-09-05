import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

veriler=pd.read_csv("loanUygun.csv")

#IMPUTE
#gerek duyulmamıştır

#ENCODE
cinsiyet=veriler.iloc[:,1:2].values
evli=veriler.iloc[:,2:3].values
egitim=veriler.iloc[:,4:5].values
serbestMeslek=veriler.iloc[:,5:6].values
yerlesim=veriler.iloc[:,-2:-1].values
loan=veriler.iloc[:,-1:].values
lbe=LabelEncoder()
cinsiyet[:,0]=lbe.fit_transform(cinsiyet.ravel())
evli[:,0]=lbe.fit_transform(evli.ravel())
egitim[:,0]=lbe.fit_transform(egitim.ravel())
serbestMeslek[:,0]=lbe.fit_transform(serbestMeslek.ravel())
yerlesim[:,0]=lbe.fit_transform(yerlesim.ravel())
loan[:,0]=lbe.fit_transform(loan.ravel())
ohe=OneHotEncoder()
cinsiyet=ohe.fit_transform(cinsiyet).toarray()
evli=ohe.fit_transform(evli).toarray()
egitim=ohe.fit_transform(egitim).toarray()
serbestMeslek=ohe.fit_transform(serbestMeslek).toarray()
yerlesim=ohe.fit_transform(yerlesim).toarray()
loan=ohe.fit_transform(loan).toarray()

#CONCAT
cinsiyet=pd.DataFrame(data=cinsiyet,index=range(len(veriler)))
evli=pd.DataFrame(data=evli,index=range(len(veriler)))
egitim=pd.DataFrame(data=egitim,index=range(len(veriler)))
serbestMeslek=pd.DataFrame(data=serbestMeslek,index=range(len(veriler)))
yerlesim=pd.DataFrame(data=yerlesim,index=range(len(veriler)))
loan=pd.DataFrame(data=loan,index=range(len(veriler)))
ozellikler=pd.concat([cinsiyet.iloc[:,1:],evli.iloc[:,1:],veriler.iloc[:,3:4],egitim.iloc[:,1:],
                      serbestMeslek.iloc[:,1:],veriler.iloc[:,6:11],yerlesim.iloc[:,1:]],
                     axis=1)
hedef=pd.concat([loan.iloc[:,1:]])

#SPLIT
x_train,x_test,y_train,y_test=train_test_split(ozellikler.values,hedef.values,test_size=0.33,random_state=0)


#GRID ARAMA
olasiDegerler={
    "criterion":["gini","entropy"],
    "max_depth":[5,7,9],
    "min_samples_leaf":[1,2,3,4],
    "min_samples_split":[4,6,8]
}
gridAra=GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),param_grid=olasiDegerler)
gridAra.fit(x_train,y_train)
print(f"En iyi parametreler: {gridAra.best_params_}")

#MODELİN KURULUMU
dtc=gridAra.best_estimator_
y_pred=dtc.predict(x_test)

#METRİKLER
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
spec=recall_score(y_test,y_pred,pos_label=0)
f1=f1_score(y_test,y_pred)
print("Karışıklık Matrisi:")
print(f"{cm}")
print(f"Accuracy (Genel Doğru Tahmin Oranı): {acc}")
print(f"Precision (Pozitif Tahmin Edilenlerin Doğruluk Oranı): {prec}")
print(f"Recall/Sensitivity (Pozitif Olanları Doğru Tahmin Oranı): {rec}")
print(f"Specificity (Negatif Olanları Doğru Tahmin Oranı): {spec}")
print(f"F1 (Prec-Rec Karışımı): {f1}")
