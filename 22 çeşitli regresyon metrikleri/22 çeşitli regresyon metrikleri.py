import numpy as np
import pandas as pd

durum=pd.read_csv("standings.csv")

#CONCAT
goller=durum.iloc[:,7:9].values
puan=durum.iloc[:,-1:].values
sira=durum.iloc[:,0:1].values
ozellikler=pd.DataFrame(data=goller,index=range(len(durum)))
hedef=pd.DataFrame(data=puan,index=range(len(durum)))
hedef2=pd.DataFrame(data=sira,index=range(len(durum)))

#SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ozellikler.values,hedef.values,test_size=0.24,random_state=0)
x_train2,x_test2,y_train2,y_test2=train_test_split(ozellikler.values,hedef2.values,test_size=0.24,random_state=0)

#SCALE
from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
yScaler = StandardScaler()
xScaler2 = StandardScaler()
yScaler2 = StandardScaler()
xScaled = xScaler.fit_transform( x_train )
xScaled2 = xScaler2.fit_transform( x_train2 )
yScaled = yScaler.fit_transform( y_train )
yScaled2 = yScaler2.fit_transform( y_train2 )

#GRIDSEARCHCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
olasiDegerler = {
    "kernel": ["linear", "poly", "rbf"],
    "gamma": [0.01, 0.05, 0.1, 0.25, 0.5],
    "C": [0.1, 0.5, 1, 2.5, 5],
    "epsilon": [0.01, 0.05, 0.1, 0.2]
}
gridAra = GridSearchCV( estimator = SVR(), param_grid = olasiDegerler )
gridAra.fit( xScaled, yScaled.ravel() )
gridAra2 = GridSearchCV( estimator = SVR(), param_grid = olasiDegerler )
gridAra2.fit( xScaled2, yScaled2.ravel() )

from sklearn.tree import DecisionTreeRegressor
olasiDegerler2 = {
    "max_depth":[1,2,3,4,5],
    "min_samples_leaf":[1,2],
    "min_samples_split":[2,3,4]
}
gridAra3 = GridSearchCV( estimator = DecisionTreeRegressor( random_state = 0 ), param_grid = olasiDegerler2, n_jobs = 4)
gridAra3.fit( x_train, y_train )
gridAra4 = GridSearchCV( estimator = DecisionTreeRegressor( random_state = 0 ), param_grid = olasiDegerler2, n_jobs = 4)
gridAra4.fit( x_train2, y_train2 )

from sklearn.ensemble import RandomForestRegressor
olasiDegerler3 = { "n_estimators": [22,24,25,28,32] }
gridAra5 = GridSearchCV( 
    estimator = RandomForestRegressor( max_depth = 4, min_samples_leaf = 1, min_samples_split = 2 ),
    param_grid = olasiDegerler3,
)
gridAra5.fit( x_train, y_train.ravel() )
gridAra6 = GridSearchCV( 
    estimator = RandomForestRegressor( max_depth = 4, min_samples_leaf = 1, min_samples_split = 2 ),
    param_grid = olasiDegerler3,
)
gridAra6.fit( x_train2, y_train2.ravel() )

#LİNEER MODEL
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
lr2=LinearRegression()
lr2.fit(x_train2,y_train2)
y_pred2=lr2.predict(x_test2)

#POLİNOMAL MODEL
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
xPol=pr.fit_transform(x_train)
lr3=LinearRegression()
lr3.fit(xPol,y_train)
y_pred3=lr3.predict(pr.fit_transform(x_test))
pr2=PolynomialFeatures(degree=2)
xPol2=pr2.fit_transform(x_train2)
lr4=LinearRegression()
lr4.fit(xPol2,y_train2)
y_pred4=lr4.predict(pr2.fit_transform(x_test2))

#DESTEK VEKTÖRÜ
svr = gridAra.best_estimator_
y_predScaled = svr.predict( xScaler.transform( x_test ) )
y_pred5 = ( yScaler.inverse_transform( y_predScaled.reshape( -1, 1 ) ) )
svr2 = gridAra2.best_estimator_
y_predScaled2 = svr2.predict( xScaler2.transform( x_test2 ) )
y_pred6 = ( yScaler2.inverse_transform( y_predScaled2.reshape( -1, 1 ) ) )

#KARAR AĞACI
dtr = gridAra3.best_estimator_
y_pred7 = dtr.predict( x_test )
dtr2 = gridAra4.best_estimator_
y_pred8 = dtr2.predict( x_test2 )

#RASTGELE ORMAN
rfr = gridAra5.best_estimator_
y_pred9 = rfr.predict( x_test )
rfr2 = gridAra6.best_estimator_
y_pred10 = rfr2.predict( x_test2 )

#R2 SKORU
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
print("LİNEER MODEL")
print(f"Puan için R2: {r2_score(y_test,y_pred)}")
print(f"Sıra için R2: {r2_score(y_test2,y_pred2)}")
print(f"Puan için MAE: {mean_absolute_error(y_test,y_pred)}")
print(f"Sıra için MAE: {mean_absolute_error(y_test2,y_pred2)}")
print(f"Puan için RMSE: {np.sqrt(mean_squared_error(y_test,y_pred))}")
print(f"Sıra için RMSE: {np.sqrt(mean_squared_error(y_test2,y_pred2))}")
cvs=cross_val_score(estimator=lr, X=x_train, y=y_train, cv=4)   #LR'de 4 katlamalı train-test süreci
cvs2=cross_val_score(estimator=lr2, X=x_train2, y=y_train2, cv=4)
print(f"Puan için Ortalama R2: {cvs.mean()}")
print(f"Sıra için Ortalama R2: {cvs2.mean()}")
print(f"Puan için R2 Standart Sapması: {cvs.std()}")
print(f"Sıra için R2 Standart Sapması: {cvs2.std()}")
print("POLİNOMAL MODEL")
print(f"Puan için R2: {r2_score(y_test,y_pred3)}")
print(f"Sıra için R2: {r2_score(y_test2,y_pred4)}")
print(f"Puan için MAE: {mean_absolute_error(y_test,y_pred3)}")
print(f"Sıra için MAE: {mean_absolute_error(y_test2,y_pred4)}")
print(f"Puan için RMSE: {np.sqrt(mean_squared_error(y_test,y_pred3))}")
print(f"Sıra için RMSE: {np.sqrt(mean_squared_error(y_test2,y_pred4))}")
cvs3=cross_val_score(estimator=lr3, X=xPol, y=y_train, cv=4)
cvs4=cross_val_score(estimator=lr4, X=xPol2, y=y_train2, cv=4)
print(f"Puan için Ortalama R2: {cvs3.mean()}")
print(f"Sıra için Ortalama R2: {cvs4.mean()}")
print(f"Puan için R2 Standart Sapması: {cvs3.std()}")
print(f"Sıra için R2 Standart Sapması: {cvs4.std()}")
print("DESTEK VEKTÖRÜ")
print(f"Puan için R2: {r2_score(y_test,y_pred5)}")
print(f"Sıra için R2: {r2_score(y_test2,y_pred6)}")
print(f"Puan için MAE: {mean_absolute_error(y_test,y_pred5)}")
print(f"Sıra için MAE: {mean_absolute_error(y_test2,y_pred6)}")
print(f"Puan için RMSE: {np.sqrt(mean_squared_error(y_test,y_pred5))}")
print(f"Sıra için RMSE: {np.sqrt(mean_squared_error(y_test2,y_pred6))}")
cvs5=cross_val_score(estimator=svr, X=xScaled, y=yScaled.ravel(), cv=4)
cvs6=cross_val_score(estimator=svr2, X=xScaled2, y=yScaled2.ravel(), cv=4)
print(f"Puan için Ortalama R2: {cvs5.mean()}")
print(f"Sıra için Ortalama R2: {cvs6.mean()}")
print(f"Puan için R2 Standart Sapması: {cvs5.std()}")
print(f"Sıra için R2 Standart Sapması: {cvs6.std()}")
print("KARAR AĞACI")
print(f"Puan için R2: {r2_score(y_test,y_pred7)}")
print(f"Sıra için R2: {r2_score(y_test2,y_pred8)}")
print(f"Puan için MAE: {mean_absolute_error(y_test,y_pred7)}")
print(f"Sıra için MAE: {mean_absolute_error(y_test2,y_pred8)}")
print(f"Puan için RMSE: {np.sqrt(mean_squared_error(y_test,y_pred7))}")
print(f"Sıra için RMSE: {np.sqrt(mean_squared_error(y_test2,y_pred8))}")
cvs7=cross_val_score(estimator=dtr, X=x_train, y=y_train, cv=4)
cvs8=cross_val_score(estimator=dtr2, X=x_train2, y=y_train2, cv=4)
print(f"Puan için Ortalama R2: {cvs7.mean()}")
print(f"Sıra için Ortalama R2: {cvs8.mean()}")
print(f"Puan için R2 Standart Sapması: {cvs7.std()}")
print(f"Sıra için R2 Standart Sapması: {cvs8.std()}")
print("RASTGELE ORMAN")
print(f"Puan için R2: {r2_score(y_test,y_pred9)}")
print(f"Sıra için R2: {r2_score(y_test2,y_pred10)}")
print(f"Puan için MAE: {mean_absolute_error(y_test,y_pred9)}")
print(f"Sıra için MAE: {mean_absolute_error(y_test2,y_pred10)}")
print(f"Puan için RMSE: {np.sqrt(mean_squared_error(y_test,y_pred9))}")
print(f"Sıra için RMSE: {np.sqrt(mean_squared_error(y_test2,y_pred10))}")
cvs9=cross_val_score(estimator=rfr, X=x_train, y=y_train.ravel(), cv=4)
cvs10=cross_val_score(estimator=rfr2, X=x_train2, y=y_train2.ravel(), cv=4)
print(f"Puan için Ortalama R2: {cvs9.mean()}")
print(f"Sıra için Ortalama R2: {cvs10.mean()}")
print(f"Puan için R2 Standart Sapması: {cvs9.std()}")
print(f"Sıra için R2 Standart Sapması: {cvs10.std()}")