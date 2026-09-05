import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing     #encode
import statsmodels.api as sm     #ols raporu
from sklearn.model_selection import train_test_split     #split
from sklearn.preprocessing import PolynomialFeatures     #polinomal regresyon
from sklearn.linear_model import LinearRegression     #doğrusal regresyon
from sklearn.metrics import r2_score     #r-square

veriler = pd.read_csv("balikAgirlik.csv")

#ENCODE (değerleri dizi hâlinde tut, label çözümü yap, onehot çözümünü yapıp dizi hâline getir)
tür = veriler.iloc[:,0:1].values
lbe = preprocessing.LabelEncoder()
tür[:,0] = lbe.fit_transform( tür.ravel() )
ohe = preprocessing.OneHotEncoder()
tür = ohe.fit_transform( tür ).toarray()

#KUKLA DEĞİŞKEN ÇIKARIMI VE CONCAT (her şeyi dataframe hâline getir)
tür = pd.DataFrame( data = tür, index = range( len( veriler ) ), columns = [ "Br", "Pa", "Pe", "Pi", "Ro", "Sm", "Wh" ] )
sayisal = pd.DataFrame( data = veriler.iloc[:,2:].values, index = range( len( veriler ) ), columns = [ "L1", "L2", "L3", "He", "Wi" ] )
ozellikler = pd.concat( [tür.iloc[:,1:], sayisal], axis = 1 )
hedef = pd.DataFrame( data = veriler.iloc[:,1:2].values, index = range( len( veriler ) ), columns = [ "Weight" ] )
tam = pd.concat( [ozellikler, hedef], axis = 1 )

#P-VALUE KONTROLÜ (sabitleri ekle, dizi hâline getir, modeli kur, özete bak)
ozellikler = sm.add_constant( ozellikler )
ozellikler = np.array( ozellikler, dtype = float )
model = sm.OLS( hedef, ozellikler ).fit()
print( model.summary() )
#rapor sonucunda width bilgisi çıkarılmıştır
ozellikler = ozellikler[:,:-1]
ozellikler = np.array( ozellikler, dtype = float )
model = sm.OLS( hedef, ozellikler ).fit()
print( model.summary() )
#rapor sonucunda height bilgisi çıkarılmıştır
ozellikler = ozellikler[:,:-1]
ozellikler = np.array( ozellikler, dtype = float )
model = sm.OLS( hedef, ozellikler ).fit()
print( model.summary() )
#rapor sonucunda whitefish bilgisi çıkarılmıştır
ozellikler = ozellikler[:,[0,1,2,3,4,5,7,8,9]]
ozellikler = np.array( ozellikler, dtype = float )
model = sm.OLS( hedef, ozellikler ).fit()
print( model.summary() )
#rapor sonucunda perch bilgisi çıkarılmıştır
ozellikler = ozellikler[:,[0,1,3,4,5,6,7,8]]
ozellikler = np.array( ozellikler, dtype = float )
model = sm.OLS( hedef, ozellikler ).fit()
print( model.summary() )
#rapor sonucunda length3 bilgisi çıkarılmıştır
ozellikler = ozellikler[:,[0,1,2,3,4,5,6]]
ozellikler = np.array( ozellikler, dtype = float )
model = sm.OLS( hedef, ozellikler ).fit()
print( model.summary() )
#rapor sonucunda roach bilgisi çıkarılmıştır
ozellikler = ozellikler[:,[0,1,2,4,5,6]]
ozellikler = np.array( ozellikler, dtype = float )
model = sm.OLS( hedef, ozellikler ).fit()
print( model.summary() )

#TRAIN-TEST-SPLIT (4 adet dizi döner)
x_train, x_test, y_train, y_test = train_test_split( ozellikler, hedef, test_size = 0.33, random_state = 0)

#REGRESYON MODELİNİN KURULUMU (polinom hâle getir, lineer çatıda df'lerle eğit, polinom hâliyle tahmin ettir)
pr = PolynomialFeatures( degree = 3 )
xPol = pr.fit_transform ( x_train )
lr = LinearRegression()
lr.fit( xPol, y_train )
y_pred = lr.predict( pr.transform( x_test ) )
print( "Tür: Parkki, Length1: 24, Length2: 24 -> Weight: ", lr.predict( pr.transform( [[1,1,0,0,24,24]] ) ) )
print( "Tür: Bream, Length1: 32, Length2: 33 -> Weight: ", lr.predict( pr.transform( [[1,0,0,0,32,33]] ) ) )
print( "Tür: Smelt, Length1: 10, Length2: 12 -> Weight: ", lr.predict( pr.transform( [[1,0,0,1,10,12]] ) ) )
print( "Tür: Pike, Length1: 41, Length2: 42 -> Weight: ", lr.predict( pr.transform( [[1,0,1,0,41,42]] ) ) )

#R-SQUARE KONTROLÜ -> (gerçek, tahmin)
print( f"R-square: {r2_score( y_test, y_pred )}" )