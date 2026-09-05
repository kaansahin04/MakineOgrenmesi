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

#DENDROGRAM KONTROLÜ

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram( sch.linkage( ozellikler, method = "ward" ) )
plt.show()

#AGGLOMERATİVE HİYERARŞİSİNİN KURULUMU

from sklearn.cluster import AgglomerativeClustering

#dendrogram grafiğine göre en uygun k değeri = 2
agk = AgglomerativeClustering(n_clusters = 2, metric = "euclidean", linkage = "ward")
y_pred=agk.fit_predict(ozellikler)

#GÖRSELLEŞTİRME
#ozellikler'de 0 tahmin edilenlerin 1. ve 2. sütunu (hacim ve maaş) için grafik
plt.scatter( ozellikler[ y_pred == 0, 1], ozellikler[ y_pred == 0, 2], s = 100, c = "r" )
plt.scatter( ozellikler[ y_pred == 1, 1], ozellikler[ y_pred == 1, 2], s = 100, c = "g" )
plt.show()
