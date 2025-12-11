import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# Получение мешка слов и векторов признаков
count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

print("Bag of words:\nСловарь 'токен:индекс':\n", count.vocabulary_)
print("Векторы признаков:\n", bag.toarray(), "\n")


# Получение 2-граммного мешка слов и векторов признаков
ngram_count = CountVectorizer(ngram_range=(2, 2))
ngram_bag = ngram_count.fit_transform(docs)

print("N-gramm BoW:\nСловарь 'токен:индекс':\n", ngram_count.vocabulary_)
print("Векторы признаков 2-gramm:\n", ngram_bag.toarray(), "\n")


# Получение TF-IDF мешка слов и векторов признаков
tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)
print("TF - IDF:\nВекторы признаков с L2 нормировкой:\n", tfidf.fit_transform(bag).toarray().round(2))


# Нахождение TF-IDF вручную (без нормировки)
tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print(f'\nTF-IDF термина "is" без нормировки = {tfidf_is:.2f}\n')


# Получение TF-IDF мешка слов и векторов признаков без нормировки
tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
print("Векторы признаков без нормировки:\n", raw_tfidf.round(2))


# Получение TF-IDF мешка слов и векторов признаков с нормировкой вручную
l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
print("Векторы признаков с L2 нормировкой вручную:\n", l2_tfidf.round(2))
