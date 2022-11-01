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
    path_dec = "Data/op_spam_v1.4/negative_polarity/deceptive_from_MTurk"
    path_truth = "Data/op_spam_v1.4/negative_polarity/truthful_from_Web"
    # corpus_dec_train = load_files_folds(path_dec, ['fold1', 'fold2', 'fold3', 'fold4'])
    # corpus_dec_test = load_files_folds(path_dec, ['fold5'])
    # corpus_truth_train = load_files_folds(path_truth, ['fold1', 'fold2', 'fold3', 'fold4'])
    # corpus_truth_test = load_files_folds(path_truth, ['fold5'])

    path_comb = "Data/combined"
    categories = ['_truth', '_deceptive']
    corpus_comb_train = load_files(path_comb, categories=categories) # dit nog opdelen (nu is train alles en niet 4/5 bijvoorbeeld)
    #corpus_comb_test = load_files_folds(path_comb, [])

    count_vect = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    X_train_counts = count_vect.fit_transform(corpus_comb_train.data)
    X_train_features = count_vect.get_feature_names_out()
    shape = X_train_counts.shape

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    shape_tf = X_train_tf.shape

    print('ho')
