import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv("tenisOdevi.csv")
print(veriler)

#ENCODE
from sklearn.preprocessing import LabelEncoder
lbEnc=LabelEncoder()
havaLabel=veriler.iloc[:,:1].values
havaLabel[:,0]=lbEnc.fit_transform(veriler.iloc[:,:1])     #ilk sütun (hava durumu) bilgileri 1-2-0 olur
print(havaLabel)
ruzgarLabel=veriler.iloc[:,-2:-1].values
ruzgarLabel[:,0]=lbEnc.fit_transform(veriler.iloc[:,-2:-1])
print(ruzgarLabel)
oynaLabel=veriler.iloc[:,-1:].values
oynaLabel[:,0]=lbEnc.fit_transform(veriler.iloc[:,-1:])
print(oynaLabel)
from sklearn.preprocessing import OneHotEncoder
ohEnc=OneHotEncoder()      #ilk sütun (hava durumu) bilgisi alınır
havaOneHot=ohEnc.fit_transform(havaLabel).toarray()    #değerler için ayrı sütunlar -> geçerli olan 1
print(havaOneHot)
ruzgarOneHot=ohEnc.fit_transform(ruzgarLabel).toarray()
print(ruzgarOneHot)
oynaOneHot=ohEnc.fit_transform(oynaLabel).toarray()
print(oynaOneHot)

#BİRLEŞTİRME
havadurumu=pd.DataFrame(data=havaOneHot, index=range(len(veriler)), columns=["overcast","rainy","sunny"])
ruzgar=pd.DataFrame(data=ruzgarOneHot, index=range(len(veriler)), columns=["false","true"])
oyna=pd.DataFrame(data=oynaOneHot, index=range(len(veriler)), columns=["no","yes"])
#kukla değişkenler çıkarılır
ilk2=pd.concat([havadurumu.iloc[:,1:], veriler.iloc[:,1:2]], axis=1)     #hava ve sıcaklık bilgileri birleşir
print(ilk2)
son2=pd.concat([ruzgar.iloc[:,1:], oyna.iloc[:,1:]], axis=1)
print(son2)
ozellikler=pd.concat([ilk2, son2], axis=1)      #hava-sıcaklık ve rüzgâr-oynama bilgisi birleşir
print(ozellikler)
hedef=veriler.iloc[:,2:3]
print(hedef)

#SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    ozellikler, hedef, test_size=0.33, random_state=0
)

#TAHMİN
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

#P-VALUE AYARLAMASI
import statsmodels.api as sm
yeniX=sm.add_constant(ozellikler)      #tamamen 1'ler içeren yeni sütun (sabit değeri için)
yeniX=np.array(yeniX, dtype=float)
model=sm.OLS(hedef, yeniX).fit()      #her sütunun oynama sütunu üzerindeki etkisi
print(model.summary())      #etkileri raporla (p-value'ya bakılacak)
yeniXTrain,yeniXTest,yeniYTrain,yeniYTest = train_test_split(
    yeniX, hedef, test_size=0.33, random_state=0
)
yeniLR=LinearRegression()
yeniLR.fit(yeniXTrain,yeniYTrain)
yeniPred=yeniLR.predict(yeniXTest)
print(yeniPred)
#en yüksek p değeri index=4 sütununda olduğundan elenir (genelde p sınırı 0.05 olarak alınır)
yeniX=yeniX[:,[0,1,3,4,5]]
yeniX=np.array(yeniX, dtype=float)
model=sm.OLS(hedef, yeniX).fit()
print(model.summary())
yeniXTrain,yeniXTest,yeniYTrain,yeniYTest = train_test_split(
    yeniX, hedef, test_size=0.33, random_state=0
)
yeniLR=LinearRegression()
yeniLR.fit(yeniXTrain,yeniYTrain)
yeniPred=yeniLR.predict(yeniXTest)
print(yeniPred)
#en yüksek p değeri index=4 sütununda olduğundan elenir (genelde p sınırı 0.05 olarak alınır)
yeniX=yeniX[:,[0,1,2,4]]
yeniX=np.array(yeniX, dtype=float)
model=sm.OLS(hedef, yeniX).fit()
print(model.summary())
yeniXTrain,yeniXTest,yeniYTrain,yeniYTest = train_test_split(
    yeniX, hedef, test_size=0.33, random_state=0
)
yeniLR=LinearRegression()
yeniLR.fit(yeniXTrain,yeniYTrain)
yeniPred=yeniLR.predict(yeniXTest)
print(yeniPred)
#en yüksek p değeri index=4 sütununda olduğundan elenir (genelde p sınırı 0.05 olarak alınır)
yeniX=yeniX[:,[0,2,3]]
yeniX=np.array(yeniX, dtype=float)
model=sm.OLS(hedef, yeniX).fit()
print(model.summary())
yeniXTrain,yeniXTest,yeniYTrain,yeniYTest = train_test_split(
    yeniX, hedef, test_size=0.33, random_state=0
)
yeniLR=LinearRegression()
yeniLR.fit(yeniXTrain,yeniYTrain)
yeniPred=yeniLR.predict(yeniXTest)
print(yeniPred)
#en yüksek p değeri index=4 sütununda olduğundan elenir (genelde p sınırı 0.05 olarak alınır)
yeniX=yeniX[:,[0,2]]
yeniX=np.array(yeniX, dtype=float)
model=sm.OLS(hedef, yeniX).fit()
print(model.summary())
yeniXTrain,yeniXTest,yeniYTrain,yeniYTest = train_test_split(
    yeniX, hedef, test_size=0.33, random_state=0
)
yeniLR=LinearRegression()
yeniLR.fit(yeniXTrain,yeniYTrain)
yeniPred=yeniLR.predict(yeniXTest)
print(yeniPred)
#en yüksek p değeri index=4 sütununda olduğundan elenir (genelde p sınırı 0.05 olarak alınır)
yeniX=yeniX[:,[0]]
yeniX=np.array(yeniX, dtype=float)
model=sm.OLS(hedef, yeniX).fit()
print(model.summary())
yeniXTrain,yeniXTest,yeniYTrain,yeniYTest = train_test_split(
    yeniX, hedef, test_size=0.33, random_state=0
)
yeniLR=LinearRegression()
yeniLR.fit(yeniXTrain,yeniYTrain)
yeniPred=yeniLR.predict(yeniXTest)
print(yeniPred)