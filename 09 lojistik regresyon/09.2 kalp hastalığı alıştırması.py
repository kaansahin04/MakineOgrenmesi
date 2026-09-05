import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

veriler=pd.read_csv("kalpHasta.csv", na_values="NA")

#IMPUTE
imputer=SimpleImputer(missing_values=np.nan, strategy="most_frequent")
eksikler=veriler.iloc[:,:].values
eksikler=imputer.fit_transform(eksikler)
veriler.iloc[:,:]=eksikler

#ENCODE
#gerek duyulmamıştır

#CONCAT
ozellikler=pd.DataFrame(veriler.iloc[:,0:-1])
hedef=pd.DataFrame(veriler.iloc[:,-1:])

#SPLIT
x_train,x_test,y_train,y_test=train_test_split(ozellikler.values,hedef.values,test_size=0.33,random_state=0)

#SCALE
xScaler=StandardScaler()
xScTr=xScaler.fit_transform(x_train)
xScTe=xScaler.transform(x_test)

#MODELİN KURULUMU
lgr=LogisticRegression(class_weight="balanced",random_state=0)
lgr.fit(xScTr, y_train.ravel())
y_pred=lgr.predict(xScTe)

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

#EŞİK DEĞİŞİKLİĞİ
y_probs = lgr.predict_proba(xScTe)[:, 1]
y_pred_opt = (y_probs >= 0.35).astype(int)
cm=confusion_matrix(y_test,y_pred_opt)
acc=accuracy_score(y_test,y_pred_opt)
prec=precision_score(y_test,y_pred_opt)
rec=recall_score(y_test,y_pred_opt)
spec=recall_score(y_test,y_pred_opt,pos_label=0)
f1=f1_score(y_test,y_pred_opt)
print("Karmaşıklık Matrisi:")
print(f"{cm}")
print(f"Accuracy (Doğru Tahmin Oranı): {acc}")
print(f"Precision (Pozitif Tahminlerin Doğru Olma Oranı, TP'nin TP+FP'ye Oranı): {prec}")
print(f"Recall/Sensitivity (Pozitifleri Doğru Tahmin Oranı, TP'nin TP+FN'ye Oranı): {rec}")
print(f"Specificity (Negatifleri Doğru Tahmin Oranı, TN'nin FP+TN'ye Oranı): {spec}")
print(f"F1 (Prec-Rec Karışımı): {f1}")