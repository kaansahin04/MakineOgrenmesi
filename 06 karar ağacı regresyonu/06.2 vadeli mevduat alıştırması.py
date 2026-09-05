import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer     #impute
from sklearn import preprocessing     #encode
from sklearn.model_selection import train_test_split     #split
from sklearn.model_selection import GridSearchCV     #grid arama
from sklearn.tree import DecisionTreeRegressor     #karar ağacı
from sklearn.metrics import r2_score     #r-square

veriler = pd.read_csv("vadeliMevduat.csv")

#IMPUTE (stratejiyi belirle, sütunlardaki değerleri dizi hâlinde tut, o diziyle eğit, dönüştür, boşlukları o diziyle doldur)
imputer = SimpleImputer( missing_values = np.nan, strategy = "mean" )
eksikler = veriler.iloc[:,[1,6]].values
imputer = imputer.fit( eksikler )
eksikler = imputer.transform( eksikler )
veriler.iloc[:,[1,6]] = eksikler

#ENCODE (değerleri dizi hâlinde tut, label çözümü yap, onehot çözümünü yapıp dizi hâline getir)
job = veriler.iloc[:,2:3].values
marital = veriler.iloc[:,3:4].values
education = veriler.iloc[:,4:5].values
default = veriler.iloc[:,5:6].values
housing = veriler.iloc[:,7:8].values
loan = veriler.iloc[:,8:9].values
contact = veriler.iloc[:,9:10].values
month = veriler.iloc[:,11:12].values
poutcome = veriler.iloc[:,16:17].values
y = veriler.iloc[:,17:18].values
lbe = preprocessing.LabelEncoder()
job[:,0] = lbe.fit_transform( job.ravel() )
marital[:,0] = lbe.fit_transform( marital.ravel() )
education[:,0] = lbe.fit_transform( education.ravel() )
default[:,0] = lbe.fit_transform( default.ravel() )
housing[:,0] = lbe.fit_transform( housing.ravel() )
loan[:,0] = lbe.fit_transform( loan.ravel() )
contact[:,0] = lbe.fit_transform( contact.ravel() )
month[:,0] = lbe.fit_transform( month.ravel() )
poutcome[:,0] = lbe.fit_transform( poutcome.ravel() )
y[:,0] = lbe.fit_transform( y.ravel() )
ohe = preprocessing.OneHotEncoder()
job = ohe.fit_transform( job ).toarray()
marital = ohe.fit_transform( marital ).toarray()
education = ohe.fit_transform( education ).toarray()
default = ohe.fit_transform( default ).toarray()
housing = ohe.fit_transform( housing ).toarray()
loan = ohe.fit_transform( loan ).toarray()
contact = ohe.fit_transform( contact ).toarray()
month = ohe.fit_transform( month ).toarray()
poutcome = ohe.fit_transform( poutcome ).toarray()
y = ohe.fit_transform( y ).toarray()

#KUKLA DEĞİŞKEN ÇIKARIMI VE CONCAT (her şeyi dataframe hâle getir)
job = pd.DataFrame( data = job, index = range( len( veriler ) ) )
marital = pd.DataFrame( data = marital, index = range( len( veriler ) ) )
education = pd.DataFrame( data = education, index = range( len( veriler ) ) )
default = pd.DataFrame( data = default, index = range( len( veriler ) ) )
housing = pd.DataFrame( data = housing, index = range( len( veriler ) ) )
loan = pd.DataFrame( data = loan, index = range( len( veriler ) ) )
contact = pd.DataFrame( data = contact, index = range( len( veriler ) ) )
month = pd.DataFrame( data = month, index = range( len( veriler ) ) )
poutcome = pd.DataFrame( data = poutcome, index = range( len( veriler ) ) )
y = pd.DataFrame( data = y, index = range( len( veriler ) ) )
metinsel = pd.concat( [ job.iloc[:,1:], marital.iloc[:,1:], education.iloc[:,1:],
                       default.iloc[:,1:], housing.iloc[:,1:], loan.iloc[:,1:],
                       contact.iloc[:,1:], month.iloc[:,1:], poutcome.iloc[:,1:] ], axis = 1 )
sayısal = pd.concat( [ veriler.iloc[:,1:2], veriler.iloc[:,6:7], veriler.iloc[:,10:11], veriler.iloc[:,12:13],
                         veriler.iloc[:,13:14], veriler.iloc[:,14:15], veriler.iloc[:,15:16] ], axis = 1 )
ozellikler = pd.concat( [metinsel, sayısal], axis = 1 )
hedef = y.iloc[:,1:]
tam = pd.concat( [ozellikler, hedef], axis = 1 )

#TRAIN-TEST-SPLIT (4 adet dizi döner)
x_train, x_test, y_train, y_test = train_test_split( ozellikler.values, hedef.values, test_size = 0.33, random_state = 0 )

#GRIDSEARCHCV (değerleri belirle, eğit)
olasiDegerler = {
    "max_depth":[16,18,20,22,24],
    "min_samples_leaf":[1,2,3,4,5],
    "min_samples_split":[494,505,515,525],
}
gridAra = GridSearchCV( estimator = DecisionTreeRegressor( random_state = 0 ), param_grid = olasiDegerler, n_jobs = 4)
gridAra.fit( x_train, y_train )
print( f"En uygun parametreler: {gridAra.best_params_}" )

#REGRESYON MODELİNİN KURULMASI (modeli grid sonuçlarına göre kur, tahmin ettir)
dtr = gridAra.best_estimator_
y_pred = dtr.predict( x_test )

#R-SQUARE KONTROLÜ -> (gerçek, tahmin)
print( f"R-square: {r2_score( y_test, y_pred )}" )

