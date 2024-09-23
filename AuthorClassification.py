import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

def loadData(filePath):
    file = open(filePath, 'r', encoding='utf-8')
    data = []
    for line in file:
        data.append(line.strip())
    file.close()
    return data

def showConfusionMatrix(yTest, yPred, modelName, representation):
    cm = confusion_matrix(yTest, yPred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=np.unique(yTest), yticklabels=np.unique(yTest))
    plt.title(f'Matrica konfuzije za model {modelName} koristeci reprezentaciju {representation}')
    plt.ylabel('Prava vrednost')
    plt.xlabel('Predvidjena vrednost')
    plt.show()

def main():
    xTrain = loadData('Data/x_train.txt')
    yTrain = loadData('Data/y_train.txt')
    xTest = loadData('Data/x_test.txt')
    yTest = loadData('Data/y_test.txt')

    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=12)
    xTest, yTest = shuffle(xTest, yTest, random_state=12)

    representations = ['BOW', 'NGRAM_3W', 'NGRAM_5C']
    
    for representation in representations:
        print(f"\nEvauliranje koristeci reprezentaciju teksta {representation}")
        
        vectorizer = None
        if representation == 'BOW':
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1))
        elif representation == 'NGRAM_3W':
            vectorizer = CountVectorizer(analyzer='word', ngram_range=(3, 3))
        elif representation == 'NGRAM_5C':
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
        
        xTrainRep = vectorizer.fit_transform(xTrain)
        xTestRep = vectorizer.transform(xTest)
        
        xTrainSparse = csr_matrix(xTrainRep)
        xTestSparse = csr_matrix(xTestRep)
        
        print(f"\nModel: SVM")
        svcModel = SVC(kernel='linear')
        svcModel.fit(xTrainSparse, yTrain)
        yPred = svcModel.predict(xTestSparse)

        print(f"Accuracy: {accuracy_score(yTest, yPred):.3f}")
        print(f"F1 Score: {f1_score(yTest, yPred, average='weighted'):.3f}")
        print(f"Precision: {precision_score(yTest, yPred, average='weighted'):.3f}")
        print(f"Recall: {recall_score(yTest, yPred, average='weighted'):.3f}")

        showConfusionMatrix(yTest, yPred, "SVM", representation)

        print(f"\nModel: NB")

        nbModel = MultinomialNB()
        nbModel.fit(xTrainSparse, yTrain)
        yPred = nbModel.predict(xTestSparse)
        print(f"Accuracy: {accuracy_score(yTest, yPred):.3f}")
        print(f"F1 Score: {f1_score(yTest, yPred, average='weighted'):.3f}")
        print(f"Precision: {precision_score(yTest, yPred, average='weighted'):.3f}")
        print(f"Recall: {recall_score(yTest, yPred, average='weighted'):.3f}")

        showConfusionMatrix(yTest, yPred, "NB", representation)

if __name__ == "__main__":
    main()
