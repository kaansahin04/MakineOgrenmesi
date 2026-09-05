import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("maaslar2.csv")
print(veriler)

#ÖZELLİK-HEDEF AYIRIMI
x=veriler.iloc[:,2:5]
y=veriler.iloc[:,5:]
print(x)
print(y)
X=x.values
Y=y.values

#TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#ÇOKLU DOĞRUSAL REGRESYON
import statsmodels.api as sm
yeniX=sm.add_constant(x)
yeniX=np.array(yeniX,dtype=float)
model=sm.OLS(Y,yeniX).fit()
print(model.summary())
yeniX=yeniX[:,[0,1,2]]
yeniX=np.array(yeniX,dtype=float)
model=sm.OLS(Y,yeniX).fit()
print(model.summary())
yeniX=yeniX[:,[0,1]]
yeniX=np.array(yeniX,dtype=float)
model=sm.OLS(Y,yeniX).fit()
print(model.summary())
x_train2,x_test2,y_train2,y_test2=train_test_split(yeniX,y,test_size=0.33,random_state=0)
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train,y_train)
print("MLR: ",mlr.predict([[10,10,100]]))
mlr2=LinearRegression()
mlr2.fit(x_train2,y_train2)
print("MLR (Geri Eleme): ",mlr2.predict([[10,10]]))

#POLİNOMAL REGRESYON
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
xPol=pr.fit_transform(x)
lr=LinearRegression()
lr.fit(xPol,y)
print("PR: ",lr.predict(pr.fit_transform([[10,10,100]])))

#DESTEK VEKTÖRÜ REGRESYONU
from sklearn.preprocessing import StandardScaler
xScaler=StandardScaler()
yScaler=StandardScaler()
xScaled=xScaler.fit_transform(x)
yScaled=yScaler.fit_transform(y)
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
olasiDegerler={
    "kernel":["linear","poly","rbf"],
    "gamma":[0.01,0.1,0.5,1,5,"scale"],
    "C":[0.1,0.5,1,10],
    "epsilon":[0.01,0.05,0.1,0.2]
}
gridArama=GridSearchCV(estimator=SVR(),param_grid=olasiDegerler)
gridArama.fit(xScaled,yScaled.ravel())
print("SVR en uygun parametreler: ", gridArama.best_params_)
svrrEnIyi=gridArama.best_estimator_
tahminEnIyiScale=svrrEnIyi.predict(xScaler.transform([[10,10,100]]))
print("SVR: ",yScaler.inverse_transform(tahminEnIyiScale.reshape(-1,1)))

#KARAR AĞACI REGRESYONU
from sklearn.tree import DecisionTreeRegressor
olasiDegerler2={
    "max_depth":[3,5,7,10,15],
    "min_samples_leaf":[1,3,5,10,20],
    "min_samples_split":[3,5,10,20,30],
}
gridArama2=GridSearchCV(estimator=DecisionTreeRegressor(),param_grid=olasiDegerler2)
gridArama2.fit(X,Y.ravel())
print("DTR en uygun parametreler: ", gridArama2.best_params_)
dtrEnIyi=gridArama2.best_estimator_
print("DTR: ",dtrEnIyi.predict([[10,10,100]]))

#RASTGELE ORMAN REGRESYONU
from sklearn.ensemble import RandomForestRegressor
olasiDegerler3={
    "n_estimators":[3,5,10,20,30],
    "max_depth":[3,5,7,10,15],
    "min_samples_leaf":[1,3,5,10,20],
    "min_samples_split":[3,5,10,20,30],
}
gridArama3=GridSearchCV(estimator=RandomForestRegressor(),param_grid=olasiDegerler3)
gridArama3.fit(X,Y.ravel())
print("RFR en uygun parametreler: ", gridArama3.best_params_)
rfrEnIyi=gridArama3.best_estimator_
print("RFR: ",rfrEnIyi.predict([[10,10,100]]))

#R-SQUARE KARŞILAŞTIRMASI
from sklearn.metrics import r2_score
print("MLR: ",r2_score(Y,mlr.predict(X)))
print("MLR (Geri Eleme): ",r2_score(Y,mlr2.predict(yeniX)))
print("PR: ",r2_score(Y,lr.predict(pr.fit_transform(X))))
print("SVR: ",r2_score(Y,svrrEnIyi.predict(X)))
print("DTR: ",r2_score(Y,dtrEnIyi.predict(X)))
print("RFR: ",r2_score(Y,rfrEnIyi.predict(X)))