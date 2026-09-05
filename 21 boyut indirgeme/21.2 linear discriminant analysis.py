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

#LDA (gözetimli özelliktedir, sınıf ayrımını önemser ve maksimize etmeye çalışır)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA( n_components = 2 )
xLdaTr = lda.fit_transform( xScTr, y_train )    #sınıf ayrımını önemsediğinden hedef sütunuyla eğitiyoruz
xLdaTe = lda.transform( xScTe )

from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression(random_state=42)
lgr.fit(xScTr,y_train)      #LDA dönüşümü öncesi LGR eğitimi
y_pred=lgr.predict(xScTe)
lgr2=LogisticRegression(random_state=42)
lgr2.fit(xLdaTr,y_train)      #LDA dönüşümü sonrası LGR eğitimi
y_pred2=lgr2.predict(xLdaTe)

from sklearn.metrics import confusion_matrix
print("LDA dönüşümü öncesine ait Karmaşıklık Matrisi:")
print(f"{confusion_matrix(y_test,y_pred)}")
print("LDA dönüşümü sonrasına ait Karmaşıklık Matrisi:")
print(f"{confusion_matrix(y_test,y_pred2)}")
print("BEFORE / AFTER:")
print(f"{confusion_matrix(y_pred,y_pred2)}")