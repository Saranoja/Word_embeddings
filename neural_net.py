from typing import List
import re


def budget_softmax(element: float, elements: List[float]):
    return element / sum(elements)


class NeuralNetwork:
    def __init__(self):
        pass


def one_hot_encode(string: str, strings: List[str]):
    one_hot_encoded = map(lambda s: 1 if s == string else 0, strings)
    return list(one_hot_encoded)


def read_file(filepath: str):
    with open(filepath) as file:
        filtered_file = filter(lambda x: x != "", file.read().split("\n"))
        return list(filtered_file)


def preprocess(corpus):
    word_tokenizer = re.compile("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+")
    stop_words = open('english_stopwords.txt').read().split("\n")

    preprocessed_text_sentences = []
    sentences = corpus.split(".")
    for sentence in sentences:
        sentence = word_tokenizer.findall(sentence)
        sentence = filter(lambda word: word not in stop_words, sentence)
        sentence = map(lambda word: word.lower(), sentence)
        sentence = list(sentence)
        preprocessed_text_sentences.append(sentence)
    return preprocessed_text_sentences


def get_training_data(preprocessed_text_sentences):
    occurrences = {}
    for sentence in preprocessed_text_sentences:
        for word in sentence:
            if word not in occurrences:
                occurrences[word] = 1
            else:
                occurrences[word] += 1
    occurrences = sorted(list(occurrences.keys()))

    vocabulary = {}
    for i in range(len(occurrences)):
        vocabulary[occurrences[i]] = i
    vocabulary_length = len(vocabulary)

    window_size = 2
    X = []
    y = []

    for sentence in preprocessed_text_sentences:
        for center_word_index in range(len(sentence)):
            center_word_one_hot = [0 for i in range(vocabulary_length)]
            outer_word_one_hot = [0 for i in range(vocabulary_length)]

            center_word_one_hot[vocabulary[sentence[center_word_index]]] = 1
            for outer_word_index in range(center_word_index - window_size, center_word_index + window_size):
                if 0 <= outer_word_index < len(sentence) and center_word_index != outer_word_index:
                    outer_word_one_hot[vocabulary[sentence[outer_word_index]]] = 1

            X.append(center_word_one_hot)
            y.append(outer_word_one_hot)

    return X, y


preprocessed_text_sentences = preprocess("I am georgi girev. I am honoured")
for l in get_training_data(preprocessed_text_sentences):
    print(l)
