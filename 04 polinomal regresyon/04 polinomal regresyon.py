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
pr2 = PolynomialFeatures(degree=2)   #2. dereceden bir polinom oluştur
xPol = pr2.fit_transform(x)
lr2 = LinearRegression()
lr2.fit(xPol, y)
#GRAFİK
plt.scatter(x,y)
plt.plot(x,lr2.predict(xPol))
plt.show()
print(xPol)
print(lr2.predict([[1,11,121]]))   #sonraki seviyedeki maaş
print(lr2.predict(pr2.fit_transform([[11]])))   #yukarıdakinin daha kolay gösterimi

#YENİ POLİNOMAL MODELLEME
from sklearn.preprocessing import PolynomialFeatures
pr4 = PolynomialFeatures(degree=4)   #4. dereceden bir polinom oluştur
xPol2 = pr4.fit_transform(x)
lr3 = LinearRegression()
lr3.fit(xPol2, y)
#GRAFİK
plt.scatter(x,y)
plt.plot(x,lr3.predict(xPol2))
plt.show()
print(xPol2)
print(lr3.predict([[1,11,121,1331,14641]]))   #sonraki seviyedeki maaş
print(lr3.predict(pr4.fit_transform([[11]])))

#YENİ POLİNOMAL MODELLEME
from sklearn.preprocessing import PolynomialFeatures
pr6 = PolynomialFeatures(degree=6)   #6. dereceden bir polinom oluştur
xPol3 = pr6.fit_transform(x)
lr4 = LinearRegression()
lr4.fit(xPol3, y)
#GRAFİK
plt.scatter(x,y)
plt.plot(x,lr4.predict(xPol3))
plt.show()
print(xPol3)
print(lr4.predict([[1,11,121,1331,14641,161051,1771561]]))   #sonraki seviyedeki maaş
print(lr4.predict(pr6.fit_transform([[11]])))