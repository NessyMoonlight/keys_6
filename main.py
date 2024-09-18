from textblob import TextBlob
from readability import Readability
import nltk



def syllable_count(str):
    count = 0

    syllables = set("AEIOUaeiouУЕАОЭЯИЮуеаоэыяию")

    for letter in str:
        if letter in syllables:
            count = count + 1

    print(count)

def sentence_count(str):
    count = 0

    syllables = set(".")

    for letter in str:
        if letter in syllables:
            count = count + 1

    print(count)


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
f = r.flesch

print(f"Предложений: {sentence_count(n)}")

print(f"Слов: {len(b.words)}")

print(f"Слогов: {syllable_count(n)}")

print(f"Средняя длина предложения в словах {int(len(b.words)) / int(sentence_count(n))}")

print(f"Средняя длина слова в слогах {int(syllable_count(n)) / int(len(b.words))}")

print(f"Индекс удобочитаемости Флеша: {print(f.ease)}")

if round(prob_dist.prob("pos"), 2)< 0.4:
    print("Тональность текста: негативный")
elif round(prob_dist.prob("pos"), 2) > 0.6:
    print("Тональность текста: позитивный")
else:
    print("Тональность текста: нейтральный")



print(f"Объективность: {100-round(b.sentiment.subjectivity,4)*100}%")