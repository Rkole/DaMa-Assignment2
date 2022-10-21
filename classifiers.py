import numpy as np
from sklearn.model_selection import train_test_split
from dataset import train_test
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


def multinomial(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix((pred, y_test))
    return matrix
