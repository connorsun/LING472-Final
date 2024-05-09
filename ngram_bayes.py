import string
import time
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBayes:
    def __init__(self):
        self.ham_total = 0
        self.ham_freq = {}
        self.spam_total = 0
        self.spam_freq = {}
        self.spamNgrams = {}
        self.hamNgrams = {}
        self.spamNgrams_total = 0
        self.hamNgrams_total = 0
        self.n = 2

    def get_training_files(self):
        SPAM_PATH = "./enron1/spam"
        HAM_PATH = "./enron1/ham"
        # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        SPAM_FILES = [join(SPAM_PATH, file) for file in listdir(SPAM_PATH) if isfile(join(SPAM_PATH, file))]
        HAM_FILES = [join(HAM_PATH, file) for file in listdir(HAM_PATH) if isfile(join(HAM_PATH, file))]
        ham_total = 0
        ham_freq = {}
        spam_total = 0
        spam_freq = {}
        hamNgrams_total = 0
        spamNgrams_total = 0
        spamNgrams = {}
        hamNgrams = {}
        spamBuilder = []
        hamBuilder = []
        for ham_file in HAM_FILES:
            file = open(ham_file, "r", encoding="ISO-8859-1")
            build = ""
            for line in file:
                translating = str.maketrans('', '', string.punctuation)
                line = line.translate(translating)
                line = line.strip().lower()
                line = line.replace('subject', '')
                build = build + " " + line
            hamBuilder.append(build)
            file.close()
        vectorizer = CountVectorizer(ngram_range=(1, self.n))
        vectorizer.fit(hamBuilder)  # build ngram dictionary
        ngram = vectorizer.transform(hamBuilder)  # get ngram
        hamNgrams = vectorizer.vocabulary_
        for element in hamNgrams:
            hamNgrams_total += hamNgrams[element]
        for spam_file in SPAM_FILES:
            file = open(spam_file, "r", encoding="ISO-8859-1")
            build = ""
            for line in file:
                translating = str.maketrans('', '', string.punctuation)
                line = line.translate(translating)
                line = line.strip().lower()
                line = line.replace('subject', '')
                build = build + " " + line
            spamBuilder.append(build)
            file.close()
        vectorizer = CountVectorizer(ngram_range=(1, self.n))
        vectorizer.fit(spamBuilder)  # build ngram dictionary
        ngram = vectorizer.transform(spamBuilder)  # get ngram
        spamNgrams = vectorizer.vocabulary_
        for element in spamNgrams:
            spamNgrams_total += spamNgrams[element]
        self.ham_total = float(ham_total)
        self.ham_freq = ham_freq
        self.spam_total = float(spam_total)
        self.spam_freq = spam_freq
        self.spamNgrams = spamNgrams
        self.hamNgrams = hamNgrams
        self.spamNgrams_total = float(spamNgrams_total)
        self.hamNgrams_total = float(hamNgrams_total)

    def test(self):
        TEST_SPAM_PATH = "./enron2/spam"
        TEST_HAM_PATH = "./enron2/ham"
        TEST_SPAM_FILES = [join(TEST_SPAM_PATH, file) for file in listdir(TEST_SPAM_PATH) if
                           isfile(join(TEST_SPAM_PATH, file))]
        TEST_HAM_FILES = [join(TEST_HAM_PATH, file) for file in listdir(TEST_HAM_PATH) if
                          isfile(join(TEST_HAM_PATH, file))]
        correct = 0
        total = 0
        for ham_file in TEST_HAM_FILES:
            if not self.predict(ham_file):
                correct += 1
            total += 1
        for spam_file in TEST_SPAM_FILES:
            if self.predict(spam_file):
                correct += 1
            total += 1
        return float(correct) / total

    def predict(self, filename):
        file = open(filename, "r", encoding="ISO-8859-1")
        total_log_ham = np.log(self.hamNgrams_total / (self.hamNgrams_total + self.spamNgrams_total))
        total_log_spam = np.log(self.spamNgrams_total / (self.hamNgrams_total + self.spamNgrams_total))
        for line in file:
            translating = str.maketrans('', '', string.punctuation)
            line = line.translate(translating)
            line = line.strip().lower()
            line = line.replace('subject', '')
            vectorizer = CountVectorizer(ngram_range=(1, self.n))
            analyzer = vectorizer.build_analyzer()
            ngrams = analyzer(line)
            for ngram in ngrams:
                total_log_ham += np.log((self.hamNgrams.get(ngram, 0) + 1) / (self.hamNgrams_total + 2))
                total_log_spam += np.log((self.spamNgrams.get(ngram, 0) + 1) / (self.spamNgrams_total + 2))
        # print("file done")
        return total_log_spam > total_log_ham


if __name__ == "__main__":
    ts = time.time()
    for n in range(1, 6):
        nb = NaiveBayes()
        nb.n = n
        nb.get_training_files()
        accuracy = nb.test()
        print("accuracy for n = " + str(nb.n) + " " + str(accuracy))
    print("time taken " + str(time.time() - ts))
