import nltk
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords


# nltk.download('omw-1.4')
def make_dataset():
    # If truthful == 1, it is a truthful review, else its deceptive
    data_list = []

    for folder in os.listdir("data/deceptive"):
        for file in os.listdir("data/deceptive/{}".format(folder)):
            with open("data/deceptive/{}/{}".format(folder, file)) as f:
                review = f.read()
                data_list.append({"sentence": review, 'truthful': 0})

        for file in os.listdir("data/truthful/{}".format(folder)):
            with open("data/truthful/{}/{}".format(folder, file)) as f:
                review = f.read()
                data_list.append({"sentence": review, 'truthful': 1})
    dataframe = pd.DataFrame.from_records(data_list)
    return dataframe


def train_test():
    train, test = train_test_split(make_dataset(), random_state=8, train_size=0.8, test_size=0.2)
    X_train = np.array(train['sentence'])
    X_test = np.array(test['sentence'])
    y_train = np.array(train['truthful'])
    y_test = np.array(test['truthful'])

    return X_train, X_test, y_train, y_test


df = train_test()


def normalize(sentences):
    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    counter = 0
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'\S+.com', '', sentence)  # Remove sites and mail addresses
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove interpunction
        sentence = nltk.word_tokenize(sentence)
        sentence = ' '.join([wnl.lemmatize(str(words)) for words in sentence])
        sentence = [word for word in sentence if not word in stopwords.words()]
        counter += 1

        print("Done:", (counter / len(sentences)) * 100, "%")


normalize(df[0])
