import numpy as np
import pandas as pd

yorumlar = pd.read_csv("restoranYorum.csv")

import nltk  #NATURAL LANGUAGE TOOLKIT
from nltk.stem.porter import PorterStemmer   #kelimelerin kökünü alma
ps = PorterStemmer()
nltk.download("stopwords")
from nltk.corpus import stopwords   #işlevsiz kelimeleri ayıklama
import re   #REGULAR EXPRESSIONS

derleme = []   #hazır hâle getireceğimiz yorumlar dizisi
for i in range(1000):
    yorum = re.sub( "[^a-zA-Z]", " " , yorumlar["Review"][i] )   #alfabede olmayan her şeyi boşlukla değiştirme
    yorum = yorum.lower()   #metni küçük harflere çevirme ("Futbol" ve "futbol" farklı ifadelerdir)
    yorum = yorum.split()   #her yorum, kelimelerinden oluşan dizi hâline gelir
    yorum = [ ps.stem(kelime) for kelime in yorum if not kelime in set( stopwords( "english" ) ) ]
    #yorumdaki işlevsizler arasında olmayan her kelimeyi gövdelerine ayır
    yorum = " ".join(yorum)   #kelimeleri boşluklarla ayır
    derleme.append(yorum)   #hazır hâle gelen yorumu dizimize ekleme

#ÖZNİTELİK ÇIKARIMI
from sklearn.feature_extraction.text import CountVectorizer   #hangi kelimenin hangi yorumda olduğunun vektör hesabı
cv = CountVectorizer( max_features = 2000 )
#ÖZELLİK-HEDEF AYIRIMI
ozellikler = cv.fit_transform( derleme ).toarray()   #vektör dizisinden sayısal ozellikler sütunları elde etme
hedef = yorumlar.iloc[:,1].values

#SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ozellikler,hedef,test_size=0.2,random_state=0)
#MODEL EĞİTİMİ
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
#METRİKLER
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Karmaşıklık Matrisi:")
print(cm)