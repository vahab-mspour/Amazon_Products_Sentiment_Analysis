
import pandas as pd
import numpy as np
import random
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk
from numpy import asarray

from sklearn.naive_bayes import GaussianNB


from sklearn import metrics
from sklearn.metrics import confusion_matrix

# https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products
# https://github.com/MehulGoel1/Sentiment-Analysis-on-Amazon-Reviews
amazon_data_file = "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"

#dataframe of amazon reviews from csv
print('reading csv file..')
df=pd.read_csv(amazon_data_file)
print('csv file columns ... ')
print(df.columns)
#print(df['reviews.text'])

#Selecting Useful Columns
df = df[['reviews.rating' , 'reviews.text' , 'reviews.title']]
#print(df['reviews.rating'])

#removing missing values
df=df.dropna(how="any")
#ploting graph on the basis of review ratings
df["reviews.rating"].value_counts().sort_values().plot.bar()

sentiment_df= df[df["reviews.rating"].notnull()]
# df.info()

sentiment_df["polarity"] = sentiment_df["reviews.rating"]>=4 # creats a column of positive(for rating 4 or 5) and negative(for rating 3 or less) values

print(sentiment_df.head(5))

sentiment_df["polarity"] = sentiment_df["polarity"].replace([True , False] , [1 , 0])
print(sentiment_df.head(5))
sentiment_df["polarity"].value_counts().plot.bar()
# print(sentiment_df)

def get_word2vec_word_vectors_dict():
    # https: // radimrehurek.com / gensim / models / word2vec.html
    from gensim.test.utils import datapath
    from gensim.models import KeyedVectors
    wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)  # C text format
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"),binary=True)  # C binary format

def get_vector_of_words_dict():
    # load the whole embedding into memory
    vectorize_index = dict()
    f = open('glove.6B.100d.txt', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        vector = asarray(values[1:], dtype='float32')
        vectorize_index[word] = vector
    f.close()
    print('Loaded %s word vectors.' % len(vectorize_index))
    return vectorize_index

my_stopwords = list(stopwords.words("english"))
useful = ['all', 'any', 'both', 'each', 'few', 'more',
          'most', 'other', 'some', 'such', 'no', 'nor',
          'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very']
my_stopwords.append('')
for word in useful:
    my_stopwords.remove(word)

def reviews_cleaner(a_review):
    #for review in r:
    token_words=word_tokenize(a_review) # from nltk.tokenize
    # print (token_words)
    token_words=[''.join(c for c in token if c not in string.punctuation)for token in token_words]
    cleaned_words = [word for word in token_words if word.lower() not in my_stopwords] #from nltk.corpus
    return cleaned_words


glove_vector = get_vector_of_words_dict()
print("glove_vector['film'] = ", glove_vector['film'])

X_data = []
y_data = []

for i in range(len(sentiment_df["reviews.text"])):
    a_review_text = sentiment_df.iloc[i]["reviews.text"]
    a_review_class_label = sentiment_df.iloc[i]["polarity"]

    y_data.append(a_review_class_label)
    # print('a_review = \n ',a_review)
    review_vector = np.zeros(100, float)# vector of zero with lenghth of 100
    n = len(a_review_text)
    cleaned_tokenized_review = reviews_cleaner(a_review_text)
    for word in cleaned_tokenized_review:
        word = word.lower()
        if word in glove_vector:
            review_vector = np.add(glove_vector[word], review_vector)
        else:
            print(word, '   not in vocab!!!')
    # print(review_vector)
    # print(np.divide(review_vector, n))
    review_vector = review_vector / n # normalizatin by the length of vectors (result is average of vectors of all words in the review)
    # print(cleaned_tokenized_review)
    X_data.append(review_vector)
    if i>29:
        break
print('X_data', X_data)
print('y_data', y_data)

data_len = len(X_data)
#for i in range len

# shuffeling data
# geenrate random indexes
data_indexes = [i for i in range(len(X_data))]
print('data index=',data_indexes)
random.shuffle(data_indexes)
random_indexes = data_indexes
print('random_indexes=',data_indexes)
print('random_indexes=',random_indexes)
shuffeled_X_data = [X_data[i] for i in random_indexes]
shuffeled_y_data = [y_data[i] for i in random_indexes]

X_train = shuffeled_X_data[0: int(0.8*len(shuffeled_X_data))]
y_train = shuffeled_y_data[0: int(0.8*len(shuffeled_X_data))]

X_test = shuffeled_X_data[int(0.8*len(shuffeled_X_data)):]
y_test = shuffeled_y_data[int(0.8*len(shuffeled_X_data)):]

print('random_indexes=',random_indexes)
print('train data random indexes=',random_indexes[0: int(0.8*len(shuffeled_X_data))])
print('X_train=',X_train)
print('y_train=',y_train)
print('X_test=',X_test)
print('y_test=',y_test)

clf = GaussianNB()
NB_classifier  = clf.fit(X_train, y_train)

y_prediction = NB_classifier.predict(X_test)

print('y_prediction of NB_classifier = ',y_prediction)

#print('My trained Nive Bayes machine \n', NB_classifier)

### Evaluation #####

y_true = y_test

print("confusion matrix")
tn, fp, fn, tp = confusion_matrix(y_true,y_prediction).ravel()
print("tn\t", "fp\t"   , "fn\t", "tp")
print(tn,"\t", fp,"\t", fn,"\t", tp)
print(confusion_matrix(y_true,y_prediction))

print("precision_score")
print(metrics.precision_score(y_true,y_prediction))

print("accuracy_score")
print(metrics.accuracy_score(y_true,y_prediction))

print("F1_score")
print(metrics.f1_score(y_true,y_prediction))


