import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold

from importnb import Notebook

with Notebook():
    import buildresultauc #Załaduj plik buildresultauc.ipynb tak, jakby był modułem Pythona
    import myclassifier

#fileName = './data/dane_med3P.csv'
fileName = './data/dane_med6P.csv' #Nowe dane
dataset = pd.read_csv(fileName,sep='|') #Odczytanie zbioru danych

noColumn = dataset.shape[1] #Ustalenie liczby kolumn w danych
print("Liczba kolumn=",noColumn)

features = dataset.iloc[:, 1:noColumn - 1]  # Wyodrębnienie części warunkowej danych
labels = dataset.iloc[:, [noColumn - 1]]  # Wyodrębnienie kolumny decyzyjnej
labels = np.ravel(labels);

model = myclassifier.MyClassifier()

skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=1234)

labels_predicted_prob = cross_val_predict(model, features, labels, n_jobs=1, cv=skf, method='predict_proba')

buildResults = buildresultauc.BuildResults()

auc = buildResults.getResultAUC(labels_predicted_prob,labels)

print(auc)