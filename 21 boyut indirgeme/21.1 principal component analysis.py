import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("Wine.csv")

ozellikler = veriler.iloc[:,0:13].values
hedef = veriler.iloc[:,13].values

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(ozellikler,hedef,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xScTr = sc.fit_transform(x_train)
xScTe = sc.fit_transform(x_test)

#PCA (gözetimsiz özelliktedir)
from sklearn.decomposition import PCA
pca = PCA( n_components = 2 )       #13 sütunluk veri 2 sütuna indirildi
xPcaTr = pca.fit_transform( xScTr )
xPcaTe = pca.transform( xScTe )

from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression(random_state=42)
lgr.fit(xScTr,y_train)      #PCA dönüşümü öncesi LGR eğitimi
y_pred=lgr.predict(xScTe)
lgr2=LogisticRegression(random_state=42)
lgr2.fit(xPcaTr,y_train)      #PCA dönüşümü sonrası LGR eğitimi
y_pred2=lgr2.predict(xPcaTe)

from sklearn.metrics import confusion_matrix
print("PCA dönüşümü öncesine ait Karmaşıklık Matrisi:")
print(f"{confusion_matrix(y_test,y_pred)}")
print("PCA dönüşümü sonrasına ait Karmaşıklık Matrisi:")
print(f"{confusion_matrix(y_test,y_pred2)}")
print("BEFORE / AFTER:")
print(f"{confusion_matrix(y_pred,y_pred2)}")