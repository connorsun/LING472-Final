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

        vectorizer = CountVectorizer(ngram_range=(1, self.n))
        ham_file_list = []
        for ham_file in HAM_FILES:
            file = open(ham_file, "r", encoding="ISO-8859-1")
            content = file.read().replace("Subject:", " ")
            ham_file_list.append(content)
            file.close()
        vectorizer.fit(ham_file_list)  # build ngram dictionary
        ngram = vectorizer.transform(ham_file_list)  # get ngram
        hamNgrams = vectorizer.vocabulary_
        if "subject" in hamNgrams:
            del hamNgrams["subject"]
        for element in hamNgrams:
            hamNgrams_total += hamNgrams[element]
        vectorizer = CountVectorizer(ngram_range=(1, self.n))
        spam_file_list = []
        for spam_file in SPAM_FILES:
            file = open(spam_file, "r", encoding="ISO-8859-1")
            content = file.read().replace("Subject:", " ")
            spam_file_list.append(content)
            file.close()
        vectorizer.fit(spam_file_list)  # build ngram dictionary
        ngram = vectorizer.transform(spam_file_list)  # get ngram
        spamNgrams = vectorizer.vocabulary_
        if "subject" in spamNgrams:
            del spamNgrams["subject"]
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
        TEST_SPAM_FILES = []
        TEST_HAM_FILES = []
        TESTING_PATH = "./Training"
        TESTING_FOLDERS_PATHS = [f.path for f in scandir(TESTING_PATH) if f.is_dir()]
        for testing_folder_path in TESTING_FOLDERS_PATHS:
            TEST_SPAM_PATH = join(testing_folder_path, "spam")
            TEST_HAM_PATH = join(testing_folder_path, "ham")
            TEST_SPAM_FILES += [join(TEST_SPAM_PATH, file) for file in listdir(TEST_SPAM_PATH) if isfile(join(TEST_SPAM_PATH, file))]
            TEST_HAM_FILES += [join(TEST_HAM_PATH, file) for file in listdir(TEST_HAM_PATH) if isfile(join(TEST_HAM_PATH, file))]
        correct = 0
        total = 0
        for ham_file in TEST_HAM_FILES:
            file = open(ham_file, "r+", encoding="ISO-8859-1")
            content = file.read().replace("Subject:", " ")
            file.seek(0)
            file.write(content)
            file.truncate()
            file.seek(0)
            if not self.predict(file):
                correct += 1
            total += 1
            file.close()
        for spam_file in TEST_SPAM_FILES:
            file = open(spam_file, "r+", encoding="ISO-8859-1")
            content = file.read().replace("Subject:", " ")
            file.seek(0)
            file.write(content)
            file.truncate()
            file.seek(0)
            if self.predict(file):
                correct += 1
            total += 1
            file.close()
        print(total - correct)
        return float(correct)/total

    def predict(self, build):
        total_log_ham = np.log(self.hamNgrams_total / (self.hamNgrams_total + self.spamNgrams_total))
        total_log_spam = np.log(self.spamNgrams_total / (self.hamNgrams_total + self.spamNgrams_total))
        vectorizer = CountVectorizer(ngram_range=(1, self.n))
        try:
            vectorizer.fit(build)  # build ngram dictionary
        except ValueError as e:
            print(build)
            return 1
        analyzer = vectorizer.build_analyzer()
        ngrams = vectorizer.vocabulary_
        for ngram in ngrams:
            total_log_ham += np.log((self.hamNgrams.get(ngram, 0) + 1) / (self.hamNgrams_total + 2))
            total_log_spam += np.log((self.spamNgrams.get(ngram, 0) + 1) / (self.spamNgrams_total + 2))
        return total_log_spam >= total_log_ham


if __name__ == "__main__":
    ts = time.time()
    for n in range(2, 5):
        nb = NgramBayes(n)
        nb.train()
        accuracy = nb.test()
        print("accuracy for n = " + str(nb.n) + " " + str(accuracy))
    print("time taken " + str(time.time() - ts))