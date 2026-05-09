import nltk
import random

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Завантаження необхідних ресурсів
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Стемер та стоп-слова
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Попередня обробка тексту
def preprocess(text):
    tokens = word_tokenize(text.lower())

    filtered_words = []

    for word in tokens:
        if word.isalpha() and word not in stop_words:
            stemmed_word = stemmer.stem(word)
            filtered_words.append(stemmed_word)

    return " ".join(filtered_words)

# Завантаження даних
documents = []

for category in movie_reviews.categories():

    for fileid in movie_reviews.fileids(category):

        text = movie_reviews.raw(fileid)

        processed_text = preprocess(text)

        documents.append((processed_text, category))

# Перемішування даних
random.shuffle(documents)

# Поділ текстів та міток
texts = [doc[0] for doc in documents]
labels = [doc[1] for doc in documents]

# Перетворення тексту у числовий формат
vectorizer = CountVectorizer(max_features=3000)

X = vectorizer.fit_transform(texts).toarray()

# Поділ на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)

# Створення моделі
model = LogisticRegression(max_iter=1000)

# Навчання моделі
model.fit(X_train, y_train)

# Передбачення
y_pred = model.predict(X_test)

# Обчислення точності
accuracy = accuracy_score(y_test, y_pred)

print("Точність моделі:", accuracy)

# Перевірка на власних прикладах
examples = [
    "This movie was amazing and interesting",
    "The film was boring and terrible"
]

for review in examples:

    processed_review = preprocess(review)

    vector = vectorizer.transform([processed_review]).toarray()

    prediction = model.predict(vector)

    print("Текст:", review)
    print("Результат:", prediction[0])
    print()
