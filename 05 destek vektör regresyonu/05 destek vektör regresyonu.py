import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("maaslar.csv")
print(veriler)

#ÖZELLİK-HEDEF AYIRIMI
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#DOĞRUSAL MODELLEME
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
#GRAFİK
plt.scatter(x,y)
plt.plot(x,lr.predict(x))
plt.show()
print(x)
print(lr.predict([[11]]))   #sonraki seviyedeki maaş

#POLİNOMAL MODELLEME
from sklearn.preprocessing import PolynomialFeatures
pr2 = PolynomialFeatures(degree=4)   #4. dereceden bir polinom oluştur
xPol = pr2.fit_transform(x)
lr2 = LinearRegression()
lr2.fit(xPol, y)
#GRAFİK
plt.scatter(x,y)
plt.plot(x,lr2.predict(xPol))
plt.show()
print(xPol)
print(lr2.predict([[1,11,121,1331,14641]]))   #sonraki seviyedeki maaş
print(lr2.predict(pr2.fit_transform([[11]])))   #yukarıdakinin daha kolay gösterimi

#ÖLÇEKLEME (SVR YAPISINDA SCALE İŞLEMİ ÖNEMLİ)
from sklearn.preprocessing import StandardScaler
stScale=StandardScaler()
xScaled=stScale.fit_transform(x)
stScale2=StandardScaler()
yScaled=stScale2.fit_transform(y)

#DESTEK VEKTÖR MAKİNESİ

from sklearn.svm import SVR

svrr=SVR(kernel="rbf")      #radial-basis fonksiyonu ile çizgi oluşturulacak
svrr.fit(xScaled, yScaled)
#GRAFİK
plt.scatter(xScaled,yScaled)
plt.plot(xScaled,svrr.predict(xScaled))
plt.show()
print(xScaled)
tahminScale=svrr.predict(stScale.transform([[11]]))     #tahmin için değeri scale hâle dönüştür
print(stScale2.inverse_transform(tahminScale.reshape(-1,1)))    #scale hâldeki değerin orijinalini al

#EN UYGUN PARAMETRELERİ ELDE ETME

from sklearn.model_selection import GridSearchCV

olasiDegerler={
    "kernel":["linear","poly","rbf"],
    "gamma":[0.01,0.1,0.5,1,5,"scale"],
    "C":[0.1,0.5,1,10],
    "epsilon":[0.01,0.05,0.1,0.2]
}   #aranmasını istediğimiz değerler
gridArama = GridSearchCV(estimator = SVR(), param_grid = olasiDegerler)
gridArama.fit(xScaled, yScaled.ravel())    #scale hâldeki x ve y için arama yapılır
print("En uygun parametreler: ", gridArama.best_params_)    #en uygun parametreler
svrrEnIyi=gridArama.best_estimator_     #en uygun parametrelerle otomatik olarak oluşturulan model
tahminEnIyiScale=svrrEnIyi.predict(stScale.transform([[11]]))
print(stScale2.inverse_transform(tahminEnIyiScale.reshape(-1,1)))
plt.scatter(xScaled,yScaled)
plt.plot(xScaled,svrrEnIyi.predict(xScaled))
plt.show()