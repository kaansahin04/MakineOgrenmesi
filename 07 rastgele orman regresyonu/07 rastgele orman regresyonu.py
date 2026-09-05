import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("maaslar.csv")
print(veriler)

#ÖZELLİK-HEDEF AYIRIMI
ozellikler=veriler.iloc[:,1:2]
hedef=veriler.iloc[:,-1:]
X=ozellikler.values
Y=hedef.values

#RASTGELE ORMAN

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators = 10, random_state=0)
rfr.fit(X,Y)
#GRAFİK
plt.scatter(X,Y)
plt.plot(X,rfr.predict(X))
plt.show()
print(rfr.predict([[5.5]]))
