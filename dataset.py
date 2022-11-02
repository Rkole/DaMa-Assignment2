import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
# Run one time:
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('wordnet')


def make_dataset():
    data_list = []

    for folder in os.listdir("data/deceptive"):
        for file in os.listdir("data/deceptive/{}".format(folder)):
            with open("data/deceptive/{}/{}".format(folder, file)) as f:
                review = f.read()
                data_list.append({"sentence": review, 'truthful': 0})

    for folder in os.listdir("data/truthful"):
        for file in os.listdir("data/truthful/{}".format(folder)):
            with open("data/truthful/{}/{}".format(folder, file)) as f:
                review = f.read()
                data_list.append({"sentence": review, 'truthful': 1})

    dataframe = pd.DataFrame.from_records(data_list)
    return dataframe


def normalize(sentences):
    wnl = WordNetLemmatizer()
    documents = []

    for sentence in sentences:
        sentence = sentence.lower()  # Make the sentence lowercase

        sentence = re.sub(r'\S+.com', '', sentence)  # Remove sites and mail addresses
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove interpunction
        sentence = re.sub(r'[0-9]', " ", sentence)  # Remove numeric values

        document = sentence.split()
        document = [wnl.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    return documents


all_data = make_dataset()
X = all_data.iloc[:, 0]
y = all_data.iloc[:, 1]
X_norm = normalize(X)

for i in [5000, 10000, 15000, 25000, 50000, 75000]:
    vectorizer = CountVectorizer(max_features=i, stop_words=stopwords.words('english'), ngram_range=(2, 2))
    X = vectorizer.fit_transform(X_norm).toarray()
    X = TfidfTransformer().fit_transform(X).toarray()

    penalty_logreg = ['l1', 'l2']

    alpha_list = [0.1, 1, 10]

    highest_score = 0
    for i in penalty_logreg:
        clf = LogisticRegressionCV(penalty=i, solver='liblinear', max_iter=100000)
        result = cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()
        if result > highest_score:
            highest_score = result
            best_parameter = i

    print(highest_score, best_parameter)
    highest_score = 0

    for alpha in alpha_list:
        clf = MultinomialNB(alpha=alpha)
        result = cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()
        if result > highest_score:
            highest_score = result
            best_parameter = alpha

    print(highest_score, best_parameter)

    tree1 = DecisionTreeClassifier()
    distribution_tree = dict(criterion=['gini', 'entropy'],
                             max_depth=[x for x in range(10, 100, 10)],
                             min_samples_split=[x for x in range(2, 10)],
                             min_samples_leaf=[x for x in range(1, 5)])

    forest1 = RandomForestClassifier()
    distribution_forest = dict(criterion=['gini', 'entropy'],
                               max_depth=[x for x in range(10, 100, 10)],
                               min_samples_split=[x for x in range(2, 10)],
                               min_samples_leaf=[x for x in range(1, 5)],
                               max_features=['sqrt', 'log2'],
                               n_estimators=[x for x in range(50, 150, 10)])


    def best_param(clf, dist, n_iter=35):
        clf = RandomizedSearchCV(clf, dist, n_iter=n_iter, cv=10)
        search = clf.fit(X, y)
        print(search.best_params_)


    best_param(tree1, distribution_tree)
    best_param(forest1, distribution_forest)

