import nltk
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from collections import Counter


# Run one time:
# nltk.download('omw-1.4')


def make_dataset():
    """
    Make a dataframe containing all sentences and classifications.
    :return: Dataframe
    """
    # If truthful == 1, it is a truthful review, else it is deceptive
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
    """
    Split the dataset into training and testing sets
    :return:
    """
    train, test = train_test_split(make_dataset(), random_state=8, train_size=0.8, test_size=0.2)
    X_train = np.array(train['sentence'])
    X_test = np.array(test['sentence'])
    y_train = np.array(train['truthful'])
    y_test = np.array(test['truthful'])

    return X_train, X_test, y_train, y_test


datas = train_test()


def normalize(sentences):
    """

    :param sentences: dataframe with sentences
    :return: dataframe with updated sentences
    """
    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    counter = 0
    new_list = []
    for sentence in sentences:
        sentence = sentence.lower()  # Make the sentence lowercase
        sentence = re.sub(r'\S+.com', '', sentence)  # Remove sites and mail addresses
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove interpunction
        sentence = re.sub(r'[0-9]', " ", sentence)  # Remove numeric values
        tokens = nltk.word_tokenize(sentence)  # Tokenize the words
        tokens = [word for word in tokens if
                  not word in stopwords.words()]  # Remove stop words (Slows down the program significantly)
        tokens = ' '.join([wnl.lemmatize(str(words)) for words in tokens])  # Lemmatize words
        tokens = nltk.word_tokenize(tokens)
        new_dict = {}
        new_dict.update({"sentence": tokens})
        new_list.append(new_dict)

        counter += 1
        print("Done:", (counter / len(sentences)) * 100, "%")
    new_df = pd.DataFrame(new_list)
    return new_df


X_train_norm = normalize(datas[0][0:2])
print(X_train_norm)

# def make_vocab(document, vocab):
#    for i in document:


#vocabulary = Counter()

# make_vocab(X_train_norm, vocabulary)

#print(vocabulary.most_common(50))
