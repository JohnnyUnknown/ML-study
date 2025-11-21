import pandas as pd
import re

# «монкипатч» (добавить функцию getargspec в inspect), чтобы pymorphy2 мог её вызывать
import inspect
if not hasattr(inspect, 'getargspec') and hasattr(inspect, 'getfullargspec'):
    def getargspec(func):
        from collections import namedtuple
        fullargspec = inspect.getfullargspec(func)
        ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
        return ArgSpec(fullargspec.args, fullargspec.varargs, fullargspec.varkw, fullargspec.defaults)
    inspect.getargspec = getargspec

import pymorphy2

# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords



morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))


#  Типичный pipeline очистки
def clean_text(text):
    # 1. Регистр
    text = text.lower()
    
    # 2. Удалить всё, кроме букв и пробелов
    text = re.sub(r'[^а-я\s]', '', text)
    
    # 3. Лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Токенизация
    tokens = text.split()
    
    # 5. Удалить стоп-слова
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # 6. Лемматизация
    tokens = [morph.parse(t)[0].normal_form for t in tokens]
    
    return ' '.join(tokens)


cleaned = clean_text("Привет!!! Я бегал в парке, и мне было очень весело :)")   # → "привет бегать парк весело"
print(cleaned)


# Функция удаления знаков препинания, кроме символов смайликов
def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text



cleaned = preprocessor("Привет!!! :) Я бегал в парке, и мне было очень весело ")   # привет я бегал в парке и мне было очень весело :)
print(cleaned)



print(preprocessor("</a>This :) is :( a test :-)!"))

df = pd.read_csv('movie_data.csv', encoding='utf-8')

df = df.rename(columns={"0": "review", "1": "sentiment"})

df['review'] = df['review'].apply(preprocessor)

