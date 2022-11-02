from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest


def load_files_folds(path, categories):
    return load_files(path, categories=categories)


def extract_features_percentage(X, y, per):
    return SelectPercentile(chi2, percentile=per).fit_transform(X, y)


def extract_features_number(X, y, num):
    return SelectKBest(chi2, k=num).fit_transform(X, y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path_dec = "data/op_spam_v1.4/negative_polarity/deceptive_from_MTurk"
    path_truth = "data/op_spam_v1.4/negative_polarity/truthful_from_Web"

    path_comb = "data/combined"
    categories = ['_truth', '_deceptive']
    corpus_comb_train = load_files(path_comb,
                                   categories=categories)  # dit nog opdelen (nu is train alles en niet 4/5 bijvoorbeeld)
    # corpus_comb_test = load_files_folds(path_comb, [])


def vectorizer(train_data, gram="bi"):
    if gram == "uni":
        ngram_range = (1, 1)
    elif gram == "bi":
        ngram_range = (2, 2)

        count_vect = CountVectorizer(analyzer='word', ngram_range=ngram_range)
        X_train_counts = count_vect.fit_transform(train_data.data)
        array1 = X_train_counts.toarray()
        X_train_features = count_vect.get_feature_names_out()
        shape = X_train_counts.shape

        tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        shape_tf = X_train_tf.shape

        print('ho')


vectorizer(corpus_comb_train, gram="bi")
