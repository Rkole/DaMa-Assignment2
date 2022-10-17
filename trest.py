import nltk
from nltk.stem import WordNetLemmatizer

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize Single Word
print(lemmatizer.lemmatize("expected"))
#> bat

print(lemmatizer.lemmatize("are"))
#> are

print(lemmatizer.lemmatize("feet"))
#> foot