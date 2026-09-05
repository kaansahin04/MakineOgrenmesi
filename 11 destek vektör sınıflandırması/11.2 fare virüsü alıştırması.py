import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

veriler=pd.read_csv("fareVirus.csv")

#IMPUTE
#gerek duyulmamıştır

#ENCODE
#gerek duyulmamıştır

#CONCAT
ozellikler=pd.DataFrame(data=veriler.iloc[:,0:2].values,index=range(len(veriler)))
hedef=pd.DataFrame(data=veriler.iloc[:,-1:].values,index=range(len(veriler)))

#SPLIT
x_train,x_test,y_train,y_test=train_test_split(ozellikler.values,hedef.values,test_size=0.33,random_state=0)

#SCALE
xScaler=StandardScaler()
xScTr=xScaler.fit_transform(x_train)
xScTe=xScaler.transform(x_test)

#GRID ARAMA
olasiDegerler={
    "kernel":["linear","poly","rbf"],
    "gamma":[0.1,0.5,1,5,"scale"],
    "C":[0.01,0.05,0.1,0.5,1]
}
gridAra=GridSearchCV(estimator=SVC(),param_grid=olasiDegerler)
gridAra.fit(xScTr,y_train.ravel())
print(f"En iyi parametreler: {gridAra.best_params_}")

#MODELİN KURULUMU
svc=gridAra.best_estimator_
y_pred=svc.predict(xScTe)

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