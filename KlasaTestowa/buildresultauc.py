import numpy as np
from sklearn import metrics

class BuildResults:


    def getResultAUC(self,labels_predicted_prob,labels):

        classes = np.sort(np.unique(labels))
        selectedDecValIndex = -1
        for i in range(0, len(classes)):
            if str(classes[i]) == str(1):
                selectedDecValIndex = i
                break

        if selectedDecValIndex == -1:
            print("Nie ma wartości decyzji")
            exit()

        labels_test_auc = []
        labels_predicted_prob_auc = []

        for i in range(0, len(labels_predicted_prob)):

            gen_prob = labels_predicted_prob[i][selectedDecValIndex]

            labels_test_auc.append(labels[i])
            labels_predicted_prob_auc.append(gen_prob)

        auc = metrics.roc_auc_score(labels_test_auc, labels_predicted_prob_auc)


        '''
        from sklearn.metrics import roc_curve, roc_auc_score
        import matplotlib.pyplot as plt

        # y_true – etykiety prawdziwe (0 lub 1)
        # y_score – prawdopodobieństwa klasy pozytywnej (np. z predict_proba[:, 1])

        fpr, tpr, thresholds = roc_curve(labels_test_auc, labels_predicted_prob_auc)
        auc_score = roc_auc_score(labels_test_auc, labels_predicted_prob_auc)


        print("fpr",fpr)
        print("tpr",tpr)

        # Wykres ROC
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], 'k--', label="Losowy klasyfikator")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()
        '''

        #print("AUC=", auc)

        return auc
