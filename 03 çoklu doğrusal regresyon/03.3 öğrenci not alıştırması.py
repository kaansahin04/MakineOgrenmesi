import numpy as np
import pandas as pd
from sklearn import preprocessing     #encode
import statsmodels.api as sm     #ols raporu
from sklearn.model_selection import train_test_split     #split
from sklearn.linear_model import LinearRegression     #doğrusal regresyon
from sklearn.metrics import r2_score     #r-square

veriler = pd.read_csv("ogrenciNot.csv")

#ENCODE (değerleri dizi hâlinde tut, label çözümü yap, onehot çözümünü yapıp dizi hâline getir)
extra = veriler.iloc[:,2:3].values
lbe = preprocessing.LabelEncoder()
extra[:,0] = lbe.fit_transform( veriler.iloc[:,2:3].values.ravel() )
ohe = preprocessing.OneHotEncoder()
extra = ohe.fit_transform( extra ).toarray()

#KUKLA DEĞİŞKEN ÇIKARIMI VE CONCAT (her şeyi dataframe hâle getir)
extra = pd.DataFrame( data = extra, index = range( len( veriler ) ), columns = ["No", "Yes"] )
ilk3 = pd.concat( [ veriler.iloc[:,:2], extra.iloc[:,1] ], axis = 1 )
ozellikler = pd.concat( [ ilk3, veriler.iloc[:,3:-1] ], axis = 1 )
hedef = pd.DataFrame( data = veriler.iloc[:,-1:].values, index = range( len( veriler ) ), columns = ["Not"] )
tam = pd.concat( [ ozellikler, veriler.iloc[:,-1:] ], axis = 1 )

#P-VALUE KONTROLÜ (sabitleri ekle, dizi hâline getir, modeli kur, özete bak)
ozellikler = sm.add_constant( ozellikler )
ozellikler = np.array( ozellikler, dtype=float )
model = sm.OLS( hedef, ozellikler ).fit()
print( model.summary() )
#rapor sonucunda herhangi bir sütun çıkarımı olmayacağı anlaşılmıştır

#TRAIN-TEST SPLIT (4 adet dizi döner)
x_train, x_test, y_train, y_test = train_test_split( ozellikler, hedef, test_size = 0.33, random_state = 0 )

#REGRESYON MODELİNİN KURULUMU (kur, dizilerle eğit, tahmin ettir)
mlr = LinearRegression()
mlr.fit( x_train, y_train )
y_pred = mlr.predict( x_test )
print( "Çalışma: 7 Saat, Vize: 70, Kurs: Yok, Uyku: 8 Saat, Deneme: 3 Tane -> Final: ", mlr.predict( [[1,7,70,0,8,3]] ) )

#R-SQUARE KONTROLÜ -> (gerçek, tahmin)
print( f"R-square: {r2_score( y_test, y_pred )}" )