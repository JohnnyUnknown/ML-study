import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind


stop = stopwords.words('english')

def tokenizer(text):
    """Очищает необработанные текстовые данные и разбивает их на однословные токены."""
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub(r'[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    """Функция-генератор, которая считывает и возвращает один документ за раз."""
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


# print(next(stream_docs(path='movie_data.csv')))


def get_minibatch(doc_stream, size):
    """Функция получает поток документов и возвращает нужное количество 
        документов, заданное параметром size."""
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# Векторизатор (вместо CountVectorizer), который не хранит все данные в памяти 
# и использует прием хэширования с помощью 32-битной функции MurmurHash3
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)


clf = SGDClassifier(loss='log_loss', random_state=1)

doc_stream = stream_docs(path='movie_data.csv')


pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for i in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')

clf = clf.partial_fit(X_test, y_test)