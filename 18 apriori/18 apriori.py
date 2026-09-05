import numpy as np
import pandas as pd

veriler=pd.read_csv("sepet.csv", header=None)   #başlıksız csv

#HER SATIRI AYRI LİSTELER HÂLİNDE TUTMA

transactions = []
for satir in range(0,7501):
    transactions.append( [ str( veriler.values[satir, urun] ) for urun in range(0,20) ] )
        
#OLUŞTURULMUŞ APYORİ KÜTÜPHANESİNİ DAHİL ETME

from apyori import apriori

kurallar = apriori( transactions, min_support = 0.01, min_confidence = 0.2, min_lift = 3, max_length = 5)
print(list(kurallar))