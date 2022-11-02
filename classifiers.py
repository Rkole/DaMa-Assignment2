import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import Bunch

def multinomial(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix((pred, y_test))
    return matrix


def logreg(X_train, X_test, y_train, y_test):
    clf = LogisticRegressionCV(penalty=12,  # Change these variables!!
                               Cs=10)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix((pred, y_test))
    return matrix


def tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix((pred, y_test))
    return matrix


def forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix((pred, y_test))
    return matrix
