import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing     #encode
from sklearn.model_selection import train_test_split     #split
from sklearn.preprocessing import StandardScaler     #scale
from sklearn.model_selection import GridSearchCV     #grid arama
from sklearn.svm import SVR     #svr

veriler = pd.read_csv("kanser.csv")

ozellikler = veriler.iloc[:,2:]
hedef = veriler.iloc[:,1:2]

#ENCODE (değerleri dizi hâlinde tut, label çözümü yap, onehot çözümünü yapıp dizi hâline getir)
hedef = hedef.values
lbe = preprocessing.LabelEncoder()
hedef[:,0] = lbe.fit_transform( hedef.ravel() )
ohe = preprocessing.OneHotEncoder()
hedef = ohe.fit_transform( hedef ).toarray()

#KUKLA DEĞİŞKEN ÇIKARIMI VE CONCAT (her şeyi dataframe hâle getir)
hedef = pd.DataFrame( data = hedef, index = range( len( veriler ) ), columns = [ "B", "M" ] )
hedef = hedef.iloc[:,0:1]

#TRAIN-TEST-SPLIT (4 adet dizi döner)
x_train, x_test, y_train, y_test = train_test_split( ozellikler, hedef, test_size = 0.33, random_state = 0 )

#SCALE (x ve y df'lerini ölçekle)
xScaler = StandardScaler()
yScaler = StandardScaler()
xScaled = xScaler.fit_transform( x_train )
yScaled = yScaler.fit_transform( y_train )

#GRIRDSEARCHCV (istenen model için değerleri belirle, eğit)
olasiDegerler = {
    "kernel": ["linear", "poly", "rbf"],
    "gamma": [0.01, 0.1, 0.5, 1, 5],
    "C": [0.1, 1, 5, 10],
    "epsilon": [0.01, 0.05, 0.1, 0.2]
}
gridAra = GridSearchCV( estimator = SVR(), param_grid = olasiDegerler )
gridAra.fit( xScaled, yScaled.ravel() )
print( f"En uygun parametreler: {gridAra.best_params_}" )

#REGRESYON MODELİNİN KURULUMU (modeli kur, tahmin ettir, tahminlerin ölçeklemesini geri al)
svrr = gridAra.best_estimator_
y_predScaled = svrr.predict( xScaler.transform( x_test ) )
y_pred = ( yScaler.inverse_transform( y_predScaled.reshape( -1, 1 ) ) )