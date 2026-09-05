import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

veriler=pd.read_excel("Iris.xls")

#IMPUTE
#gerek duyulmamıştır

#ENCODE
yaprak=veriler.iloc[:,-1:].values
lbe=LabelEncoder()
yaprak=lbe.fit_transform(yaprak.ravel())

#CONCAT
yaprak=pd.DataFrame(data=yaprak,index=range(len(veriler)))
ozellikler=pd.DataFrame(data=veriler.iloc[:,0:-1],index=range(len(veriler)))
hedef=pd.DataFrame(data=yaprak,index=range(len(veriler)))

#SPLIT
x_train,x_test,y_train,y_test=train_test_split(ozellikler.values,hedef.values,test_size=0.33,random_state=0)

#SCALE
scaler=StandardScaler()
xScTr=scaler.fit_transform(x_train)
xScTe=scaler.fit_transform(x_test)

#GRID ARAMA
olasiDegerler3={
    "kernel":["linear","poly","rbf"],
    "gamma":[0.04,0.045,0.05,0.055,"scale","auto"],
    "C":[4,5,6,7,8]
}
gridAra3=GridSearchCV(estimator=SVC(),param_grid=olasiDegerler3,n_jobs=4)
gridAra3.fit(xScTr,y_train.ravel())

olasiDegerler5={
    "criterion":["gini","entropy"],
    "max_depth":[1,2,3,4],
    "min_samples_leaf":[1,2,3],
    "min_samples_split":[2,3,4]
}
gridAra5=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=olasiDegerler5,n_jobs=4)
gridAra5.fit(x_train,y_train.ravel())

olasiDegerler6={
    "n_estimators":[60,65,70,75],
    "criterion":["gini","entropy"],
    "max_depth":[1,2,3],
    "min_samples_leaf":[2,3,4],
    "min_samples_split":[4,5,6]
}
gridAra6=GridSearchCV(estimator=RandomForestClassifier(),param_grid=olasiDegerler6,n_jobs=-1)
gridAra6.fit(x_train,y_train.ravel())

#LOJİSTİK REGRESYON
lgr=LogisticRegression(random_state=0)
lgr.fit(xScTr,y_train.ravel())
y_pred=lgr.predict(xScTe)

#METRİKLER
cm1=confusion_matrix(y_test,y_pred)
acc1=accuracy_score(y_test,y_pred)
prec1=precision_score(y_test,y_pred,average="weighted")
rec1=recall_score(y_test,y_pred,average="weighted")
f11=f1_score(y_test,y_pred,average="weighted")
print("=========LOJİSTİK REGRESYON=========")
print("Karmaşıklık Matrisi:")
print(f"{cm1}")
print(f"Accuracy (Genel Doğru Tahmin Oranı): {acc1}")
print(f"Precision (Yapılan Tahminin Doğru Çıkma Oranı): {prec1}")
print(f"Recall/Sensitivity (Değerleri Doğru Tahmin Etme Oranı): {rec1}")
print(f"F1 (Precision-Recall Karışımı): {f11}")

#KNN
knn=KNeighborsClassifier(n_neighbors=10,weights="uniform",metric="manhattan")
knn.fit(xScTr,y_train.ravel())
y_pred2=knn.predict(xScTe)

#METRİKLER
cm2=confusion_matrix(y_test,y_pred2)
acc2=accuracy_score(y_test,y_pred2)
prec2=precision_score(y_test,y_pred2,average="weighted")
rec2=recall_score(y_test,y_pred2,average="weighted")
f12=f1_score(y_test,y_pred2,average="weighted")
print()
print("=========KNN=========")
print("Karmaşıklık Matrisi:")
print(f"{cm2}")
print(f"Accuracy (Genel Doğru Tahmin Oranı): {acc2}")
print(f"Precision (Yapılan Tahminin Doğru Çıkma Oranı): {prec2}")
print(f"Recall/Sensitivity (Değerleri Doğru Tahmin Etme Oranı): {rec2}")
print(f"F1 (Precision-Recall Karışımı): {f12}")

#DESTEK VEKTÖRÜ
svc=gridAra3.best_estimator_
y_pred3=svc.predict(xScTe)

#METRİKLER
cm3=confusion_matrix(y_test,y_pred3)
acc3=accuracy_score(y_test,y_pred3)
prec3=precision_score(y_test,y_pred3,average="weighted")
rec3=recall_score(y_test,y_pred3,average="weighted")
f13=f1_score(y_test,y_pred3,average="weighted")
print()
print(f"=========DESTEK VEKTÖRÜ========= (En iyi parametreler: {gridAra3.best_params_})")
print("Karmaşıklık Matrisi:")
print(f"{cm3}")
print(f"Accuracy (Genel Doğru Tahmin Oranı): {acc3}")
print(f"Precision (Yapılan Tahminin Doğru Çıkma Oranı): {prec3}")
print(f"Recall/Sensitivity (Değerleri Doğru Tahmin Etme Oranı): {rec3}")
print(f"F1 (Precision-Recall Karışımı): {f13}")

#NAIVE BAYES
gnb=GaussianNB()
gnb.fit(x_train,y_train.ravel())
y_pred4=gnb.predict(x_test)

#METRİKLER
cm4=confusion_matrix(y_test,y_pred4)
acc4=accuracy_score(y_test,y_pred4)
prec4=precision_score(y_test,y_pred4,average="weighted")
rec4=recall_score(y_test,y_pred4,average="weighted")
f14=f1_score(y_test,y_pred4,average="weighted")
print()
print(f"=========GAUSSIAN NAIVE BAYES=========")
print("Karmaşıklık Matrisi:")
print(f"{cm4}")
print(f"Accuracy (Genel Doğru Tahmin Oranı): {acc4}")
print(f"Precision (Yapılan Tahminin Doğru Çıkma Oranı): {prec4}")
print(f"Recall/Sensitivity (Değerleri Doğru Tahmin Etme Oranı): {rec4}")
print(f"F1 (Precision-Recall Karışımı): {f14}")

#KARAR AĞACI
dtc=gridAra5.best_estimator_
y_pred5=dtc.predict(x_test)

#METRİKLER
cm5=confusion_matrix(y_test,y_pred5)
acc5=accuracy_score(y_test,y_pred5)
prec5=precision_score(y_test,y_pred5,average="weighted")
rec5=recall_score(y_test,y_pred5,average="weighted")
f15=f1_score(y_test,y_pred5,average="weighted")
print()
print(f"=========KARAR AĞACI========= (En iyi parametreler: {gridAra5.best_params_})")
print("Karmaşıklık Matrisi:")
print(f"{cm5}")
print(f"Accuracy (Genel Doğru Tahmin Oranı): {acc5}")
print(f"Precision (Yapılan Tahminin Doğru Çıkma Oranı): {prec5}")
print(f"Recall/Sensitivity (Değerleri Doğru Tahmin Etme Oranı): {rec5}")
print(f"F1 (Precision-Recall Karışımı): {f15}")

#RASTGELE ORMAN
rfc=gridAra6.best_estimator_
y_pred6=rfc.predict(x_test)

#METRİKLER
cm6=confusion_matrix(y_test,y_pred6)
acc6=accuracy_score(y_test,y_pred6)
prec6=precision_score(y_test,y_pred6,average="weighted")
rec6=recall_score(y_test,y_pred6,average="weighted")
f16=f1_score(y_test,y_pred6,average="weighted")
print()
print(f"=========RASTGELE ORMAN========= (En iyi parametreler: {gridAra6.best_params_})")
print("Karmaşıklık Matrisi:")
print(f"{cm6}")
print(f"Accuracy (Genel Doğru Tahmin Oranı): {acc6}")
print(f"Precision (Yapılan Tahminin Doğru Çıkma Oranı): {prec6}")
print(f"Recall/Sensitivity (Değerleri Doğru Tahmin Etme Oranı): {rec6}")
print(f"F1 (Precision-Recall Karışımı): {f16}")

#ROC GRAFİĞİ
plt.scatter(0.9,97.9,c="b",marker="o",s=50)
plt.scatter(2.2,95.6,c="g",marker="o",s=50)
plt.scatter(2,96,c="r",marker="o",s=50)
plt.plot(np.arange(0,100),np.arange(0,100),c="black")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Grafiği")

#Kazanan Sınıflandırma Algoritmaları: Lojistik Regresyon, KNN (K-En Yakın Komşu), Destek Vektörü, Karar Ağacı.