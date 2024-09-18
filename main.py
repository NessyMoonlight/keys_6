# Case-study #6
# Developers: Borodin Artemiy, Solovyova Maria,
# Selikhova Polina
#

import ru_local as ru

from textblob import TextBlob
from readability import Readability


def syllable_count(text):
    count = 0
    syllables = set("AEIOUaeiouУЕАОЭЯИЮуеаоэыяию")

    for letter in text:
        if letter in syllables:
            count += 1

    return count


def sentence_count(text):
    return len([s for s in text.split('.') if s.strip()])


train = [
    ("I love this sandwich.", "pos"),
    ("this is an amazing place!", "pos"),
    ("I feel very good about these beers.", "pos"),
    ("this is my best work.", "pos"),
    ("what an awesome view", "pos"),
    ("I do not like this restaurant", "neg"),
    ("I am tired of this stuff.", "neg"),
    ("I can't deal with this", "neg"),
    ("he is my sworn enemy!", "neg"),
    ("my boss is horrible.", "neg"),
]
test = [
    ("the beer was good.", "pos"),
    ("I do not enjoy my job", "neg"),
    ("I ain't feeling dandy today.", "neg"),
    ("I feel amazing!", "pos"),
    ("Gary is a friend of mine.", "pos"),
    ("I can't believe I'm doing this.", "neg"),
]

from textblob.classifiers import NaiveBayesClassifier

cl = NaiveBayesClassifier(train)

n = input("Введите текст: ")
b = TextBlob(n)
cl.classify(n)
prob_dist = cl.prob_classify(n)
r = Readability(n)

sentences = sentence_count(n)
syllables = syllable_count(n)

average_sentence_length = len(b.words) / sentences if sentences > 0 else 0
average_word_length = syllables / len(b.words) if len(b.words) > 0 else 0

flesch = r.flesch

print(f"Предложений: {sentences}")
print(f"Слов: {len(b.words)}")
print(f"Слогов: {syllables}")
print(f"Средняя длина предложения в словах: {average_sentence_length}")
print(f"Средняя длина слова в слогах: {average_word_length}")
print(f"Индекс удобочитаемости Флеша: {flesch.ease}")

if round(prob_dist.prob("pos"), 2) < 0.4:
    print("Тональность текста: негативный")
elif round(prob_dist.prob("pos"), 2) > 0.6:
    print("Тональность текста: позитивный")
else:
    print("Тональность текста: нейтральный")

print(f"Объективность: {100 - round(b.sentiment.subjectivity, 4) * 100}%")
