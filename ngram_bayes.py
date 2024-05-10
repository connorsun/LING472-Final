import string
import time
from os import listdir, scandir
from os.path import isfile, join
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from naive_bayes import NaiveBayes


class NgramBayes(NaiveBayes):
    def __init__(self, n):
        self.ham_total = 0
        self.ham_freq = {}
        self.spam_total = 0
        self.spam_freq = {}
        self.spamNgrams = {}
        self.hamNgrams = {}
        self.spamNgrams_total = 0
        self.hamNgrams_total = 0
        self.n = n

    def train(self):
        # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        SPAM_FILES = []
        HAM_FILES = []
        TESTING_PATH = "./Training"
        TESTING_FOLDERS_PATHS = [f.path for f in scandir(TESTING_PATH) if f.is_dir()]
        for testing_folder_path in TESTING_FOLDERS_PATHS:
            TEST_SPAM_PATH = join(testing_folder_path, "spam")
            TEST_HAM_PATH = join(testing_folder_path, "ham")
            SPAM_FILES += [join(TEST_SPAM_PATH, file) for file in listdir(TEST_SPAM_PATH) if isfile(join(TEST_SPAM_PATH, file))]
            HAM_FILES += [join(TEST_HAM_PATH, file) for file in listdir(TEST_HAM_PATH) if isfile(join(TEST_HAM_PATH, file))]
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
        nb = NgramBayes(n)
        nb.train()
        accuracy = nb.test()
        print("accuracy for n = " + str(nb.n) + " " + str(accuracy))
    print("time taken " + str(time.time() - ts))
