import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score

#Klasa do optymalizacji parametrów klasyfikatora

class BuildResults:

    #Pojedynczy test
    def getSingleResult(self,labels_predicted_prob,labels):
        classes = np.sort(np.unique(labels))
        selectedDecValIndex = -1
        for i in range(0, len(classes)):
            if str(classes[i]) == str(1):
                selectedDecValIndex = i
                break

        if selectedDecValIndex == -1:
            print("Nie ma wartości decyzji")
            exit()

        labels_predicted = []
        labels_test = []
        for i in range(0, len(labels_predicted_prob)):

            gen_prob = labels_predicted_prob[i][selectedDecValIndex]

            oryg_dec = labels[i]

            # if gen_prob > 0.18:
            if gen_prob > 0.01:
                labels_predicted.append(3)
                labels_test.append(oryg_dec)
            else:
                labels_predicted.append(0)
                labels_test.append(oryg_dec)

        # Policzenie jakości klasyfikacji przez porównanie: labels_predicted i labels_test
        accuracy = metrics.accuracy_score(labels_test, labels_predicted)

        print("Dokładnośc klasyfikacji=", accuracy)

        print("========= PEŁNE WYNIKI KLASYFIKACJI ================")

        report = classification_report(labels_test, labels_predicted)
        print(report)


    #Optymalizacja theshold-a
    def getOptimalResult(self,labels_predicted_prob,labels):

        classes = np.sort(np.unique(labels))
        selectedDecValIndex = -1
        for i in range(0, len(classes)):
            if str(classes[i]) == str(1):
                selectedDecValIndex = i
                break

        if selectedDecValIndex == -1:
            print("Nie ma wartości decyzji")
            exit()

        THRESHOLD = 0.01

        optThereshold = 0.0
        optAccuracy = 0.0
        optRecall0 = 0.0
        optRecall1 = 0.0

        minDiff = 1.0

        while THRESHOLD < 0.99:
            labels_predicted = []
            labels_test = []
            for i in range(0, len(labels_predicted_prob)):

                gen_prob = labels_predicted_prob[i][selectedDecValIndex]

                oryg_dec = labels[i]

                if gen_prob > THRESHOLD:
                    labels_predicted.append(1)
                    labels_test.append(oryg_dec)
                else:
                    labels_predicted.append(0)
                    labels_test.append(oryg_dec)

            # Policzenie jakości klasyfikacji przez porównanie: labels_predicted i labels_test
            accuracy = metrics.accuracy_score(labels_test, labels_predicted)

            recall0 = recall_score(labels_test, labels_predicted, pos_label=0)
            recall1 = recall_score(labels_test, labels_predicted, pos_label=1)

            diff = abs(recall0 - recall1)

            TEXT = "Diff="+str(diff)+" THRESHOLD="+str(THRESHOLD)+" recall0="+str(recall0) +" recall1="+str(recall1)+" accuracy="+str(accuracy)

            if diff<minDiff:
                minDiff = diff
                optThereshold = THRESHOLD
                optAccuracy = accuracy
                optRecall0 = recall0
                optRecall1 = recall1

            THRESHOLD = THRESHOLD + 0.01


        return optThereshold, optRecall0, optRecall1, optAccuracy
