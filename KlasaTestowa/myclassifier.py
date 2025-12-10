import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

#Klasa opakowująca klasyfikator
class MyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.model = None #Miejsce na lokalny model
        self.classes_ = None #Miejsce na klasy decyzyjne


    #Uczenie klasyfikatora
    def fit(self, features_train, labels_train):

        #Tutaj jest demonstracja klasyfikatora (do środka wstawiamy las losowy, ale może to być np. sieć neuronowa)
        self.classes_ = np.sort(np.unique(labels_train))
        self.model = RandomForestClassifier(n_jobs=50, random_state=1234)
        self.model.fit(features_train, np.ravel(labels_train))  # Uczenie klasyfikatora na części treningowej


    #Testowanie klasyfikatora
    def predictProb(self, features_test):

        labels_predicted_prob = self.model.predict_proba(features_test)  # Generowania decyzji dla części testowej

        classes = self.model.classes_ #Pobieranie z modelu listy klas decyzyjnych

        #Szukamy numeru decyzji 1
        selectedDecValIndex = -1
        for i in range(0, len(classes)):
            if str(classes[i]) == str(1): #Interesuje nas decyzja 1
                selectedDecValIndex = i
                break

        if selectedDecValIndex == -1:
            print("Nie ma wartości decyzji")
            exit()

        #Utworzenie listy prawdopodobieńst dla klasy 1
        labels_prob_list = []
        for i in range(0, len(labels_predicted_prob)):
            gen_prob = labels_predicted_prob[i][selectedDecValIndex]
            labels_prob_list.append(gen_prob)


        return labels_prob_list


    #Zwrócenie par prawdopodobieństw na obydwie klasy dla wszystkich wierszy
    def predict_proba(self, features_test):

        labels_predicted_prob = self.predictProb(features_test)

        number_row = features_test.shape[0]

        #Utworzenie macierzy par (bo dwie klasy decyzyjne) wypełnionej zerami o długości liczby wierszy w danych
        pair_labels_predicted_prob = np.zeros((number_row, len(self.classes_)))

        #Wypełnianie macierzy prawdopodobieństwami
        for i in range(0, number_row):
            loc_prob_1 = float(labels_predicted_prob[i])
            loc_prob_0 = 1.0 - loc_prob_1
            pair_labels_predicted_prob[i,0] = loc_prob_0
            pair_labels_predicted_prob[i,1] = loc_prob_1

        return pair_labels_predicted_prob


    def predict(self, X):
        print("Nie zaimplementowane")
        exit()
