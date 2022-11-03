from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar

def uni_multinomial(X_train, X_test, y_train, y_test):
    clf = MultinomialNB(alpha=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix(pred, y_test)
    scores = {'accuracy': round(accuracy_score(pred, y_test), 3), 'precision': round(precision_score(pred, y_test), 3),
              'recall': round(recall_score(pred, y_test), 3), 'f1': round(f1_score(pred, y_test), 3), 'matrix': matrix}
    return scores


def bi_multinomial(X_train, X_test, y_train, y_test):
    clf = MultinomialNB(alpha=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix(pred, y_test)
    scores = {'accuracy': round(accuracy_score(pred, y_test), 3), 'precision': round(precision_score(pred, y_test), 3),
              'recall': round(recall_score(pred, y_test), 3), 'f1': round(f1_score(pred, y_test), 3), 'matrix': matrix}
    return scores

def uni_logreg(X_train, X_test, y_train, y_test):
    clf = LogisticRegressionCV(penalty='l2')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix(pred, y_test)
    scores = {'accuracy': round(accuracy_score(pred, y_test), 3), 'precision': round(precision_score(pred, y_test), 3),
              'recall': round(recall_score(pred, y_test), 3), 'f1': round(f1_score(pred, y_test), 3), 'matrix': matrix}
    return scores


def bi_logreg(X_train, X_test, y_train, y_test):
    clf = LogisticRegressionCV(penalty='l2')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix(pred, y_test)
    scores = {'accuracy': round(accuracy_score(pred, y_test), 3), 'precision': round(precision_score(pred, y_test), 3),
              'recall': round(recall_score(pred, y_test), 3), 'f1': round(f1_score(pred, y_test), 3), 'matrix': matrix}
    return scores


def uni_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(criterion='entropy',
                                 max_depth=10,
                                 min_samples_split=9,
                                 min_samples_leaf=3)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix(pred, y_test)
    scores = {'accuracy': round(accuracy_score(pred, y_test), 3), 'precision': round(precision_score(pred, y_test), 3),
              'recall': round(recall_score(pred, y_test), 3), 'f1': round(f1_score(pred, y_test), 3), 'matrix': matrix}
    return scores


def bi_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(criterion='entropy',
                                 max_depth=80,
                                 min_samples_split=7,
                                 min_samples_leaf=4)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix(pred, y_test)
    scores = {'accuracy': round(accuracy_score(pred, y_test), 3), 'precision': round(precision_score(pred, y_test), 3),
              'recall': round(recall_score(pred, y_test), 3), 'f1': round(f1_score(pred, y_test), 3), 'matrix': matrix}
    return scores


def uni_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(criterion='gini',
                                 max_depth=60,
                                 min_samples_split=5,
                                 min_samples_leaf=4,
                                 max_features='sqrt',
                                 n_estimators=110)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix(pred, y_test)
    scores = {'accuracy': round(accuracy_score(pred, y_test), 3), 'precision': round(precision_score(pred, y_test), 3),
              'recall': round(recall_score(pred, y_test), 3), 'f1': round(f1_score(pred, y_test), 3), 'matrix': matrix}
    return scores


def bi_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(criterion='entropy',
                                 max_depth=70,
                                 min_samples_split=6,
                                 min_samples_leaf=2,
                                 max_features='log2',
                                 n_estimators=100)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    matrix = confusion_matrix(pred, y_test)
    scores = {'accuracy': round(accuracy_score(pred, y_test), 3), 'precision': round(precision_score(pred, y_test), 3),
              'recall': round(recall_score(pred, y_test), 3), 'f1': round(f1_score(pred, y_test), 3), 'matrix': matrix}
    return scores
