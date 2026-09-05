import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veriler=pd.read_csv("musteri.csv")

ozellikler=veriler.iloc[:,2:].values

#SPLIT
x_train,x_test=train_test_split(ozellikler,test_size=0.33,random_state=0)

#SCALE
scaler=StandardScaler()
xScTr=scaler.fit_transform(x_train)
xScTe=scaler.transform(x_test)

#K-MEANS MODELİNİN KURULUMU

from sklearn.cluster import KMeans

wcss = []       #wcss değerlerini tutacak list
for i in range(1,11):
    kmDeneme = KMeans(n_clusters = i, init = "k-means++", random_state = 123)
    kmDeneme.fit(xScTr)
    wcss.append(kmDeneme.inertia_)    #optimum k seçimine yönelik olarak her k değeri için WCSS değeri
plt.plot(range(1,11), wcss)     #wcss grafiğinde dirsek nokta, k değeri olarak seçilebilir

km=KMeans(n_clusters=2,init="k-means++",random_state=123)
km.fit(xScTr)
print(kmDeneme.cluster_centers_)      #kümelerin merkez noktaları
y_pred=km.predict(xScTe)