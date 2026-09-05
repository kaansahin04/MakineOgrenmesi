import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("maaslar.csv")
print(veriler)

#ÖZELLİK-HEDEF AYIRIMI
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
X=x.values      #DataFrame'i numPy dizisi hâline getirdik
Y=y.values

#KARAR AĞACI

from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(X,Y)
#GRAFİK
plt.scatter(X,Y)
plt.plot(X,dtr.predict(X))
plt.show()
print(dtr.predict([[11]]))