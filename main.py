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

n = input(f"{ru.TEXT}")
b = TextBlob(n)
cl.classify(n)
prob_dist = cl.prob_classify(n)
r = Readability(n)

sentences = sentence_count(n)
syllables = syllable_count(n)

average_sentence_length = len(b.words) / sentences if sentences > 0 else 0
average_word_length = syllables / len(b.words) if len(b.words) > 0 else 0

flesch = r.flesch

print(f"{ru.SENTENCE} {sentences}")
print(f"{ru.WORD} {len(b.words)}")
print(f"{ru.SYLLABLE} {syllables}")
print(f"{ru.AVERAGE_SENTENCE} {average_sentence_length}")
print(f"{ru.AVERAGE_WORD} {average_word_length}")
print(f"{ru.FLASH_INDEX} {flesch.ease}")

if round(prob_dist.prob("pos"), 2) < 0.4:
    print(f"{ru.NEGATIVE}")
elif round(prob_dist.prob("pos"), 2) > 0.6:
    print(f"{ru.POSITIVE}")
else:
    print(f"{ru.NEUTRAL}")

print(f"{ru.OBJECTIVITY} {100 - round(b.sentiment.subjectivity, 4) * 100}%")